import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

def _is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"

def _supports_host_descriptor():
    return _is_cuda() and torch.cuda.get_device_capability()[0] >= 9

DEVICE = triton.runtime.driver.active.get_active_torch_device()
assert triton.runtime.driver.active.get_current_target().backend == "cuda"
supports_host_descriptor = _supports_host_descriptor()

def _attn_fwd_pre_hook(nargs):
    BLOCK_I = nargs["BLOCK_I"]
    BLOCK_K = nargs["BLOCK_K"]
    BLOCK_J = nargs["BLOCK_J"]
    HEAD_DIM = nargs["HEAD_DIM"]
    BLOCK_HD = nargs["BLOCK_HD"]
    if not isinstance(nargs["Lq_ptrs"], TensorDescriptor):
        return
    nargs["Lq_ptrs"].block_shape = [1, 1, BLOCK_I, BLOCK_J, HEAD_DIM]
    nargs["Lk_ptrs"].block_shape = [1, 1, BLOCK_K, 1, BLOCK_HD]
    nargs["Rq_ptrs"].block_shape = [1, 1, BLOCK_J, 1, HEAD_DIM]
    nargs["Rk_ptrs"].block_shape = [1, 1, BLOCK_K, 1, BLOCK_HD]
    nargs["v_ptrs"].block_shape = [1, 1, BLOCK_K, 1, HEAD_DIM]
    nargs["o_ptrs"].block_shape = [1, 1, BLOCK_I, BLOCK_J, HEAD_DIM]

attn_fwd_configs = [
    triton.Config({'BLOCK_I': BI, 'BLOCK_K': BK, 'BLOCK_J': BJ, 'BLOCK_HD': BHD}, num_stages=s, num_warps=w, pre_hook=_attn_fwd_pre_hook) \
    for BI in [1, 2, 4, 16, 128]\
    for BK in [16, 32, 64]\
    for BJ in [1, 2, 4, 16, 32]\
    for BHD in [16, 128]\
    for s in [2, 4] \
    for w in [4, 8]\
]

def my_filter(c):
    i = c.all_kwargs()["BLOCK_I"]
    j = c.all_kwargs()["BLOCK_J"]
    k = c.all_kwargs()["BLOCK_K"]
    return i * j >= 16 and i * j * k < 2048

attn_fwd_configs = list(filter(my_filter, attn_fwd_configs))
print(len(attn_fwd_configs), "configs for attn_fwd")

# 50% sparse:
# L40 config
# attn_fwd_configs = [
#     triton.Config({'BLOCK_I': 2, 'BLOCK_K': 16, 'BLOCK_J': 32, 'BLOCK_HD': 16}, num_stages=2, num_warps=4, pre_hook=_attn_fwd_pre_hook)
# ]
# H100 config
# attn_fwd_configs = [
#     triton.Config({'BLOCK_I': 2, 'BLOCK_K': 16, 'BLOCK_J': 32, 'BLOCK_HD': 128}, num_stages=4, num_warps=4, pre_hook=_attn_fwd_pre_hook)
# ]

# 95% sparse:
# L40 config
# attn_fwd_configs = [
#     triton.Config({'BLOCK_I': 16, 'BLOCK_K': 16, 'BLOCK_J': 4, 'BLOCK_HD': 16}, num_stages=4, num_warps=4, pre_hook=_attn_fwd_pre_hook)
# ]
# H100 config
attn_fwd_configs = [
    triton.Config({'BLOCK_I': 2, 'BLOCK_K': 16, 'BLOCK_J': 32, 'BLOCK_HD': 128}, num_stages=4, num_warps=4, pre_hook=_attn_fwd_pre_hook)
]

@triton.jit
def _maybe_make_tensor_desc(desc_or_ptr, shape, strides, block_shape):
    if isinstance(desc_or_ptr, tl.tensor_descriptor):
        return desc_or_ptr
    else:
        return tl.make_tensor_descriptor(desc_or_ptr, shape, strides, block_shape)

@triton.autotune(configs=attn_fwd_configs, key=["HEAD_DIM", "NUM_HEADS", "block_b1", "block_b2"])
@triton.jit
def _attn_fwd(sm_scale, bsz,
              nframes,
              Lq_ptrs, Lk_ptrs,
              Rq_ptrs, Rk_ptrs,
              v_ptrs, o_ptrs,
              lse_ptrs,
              block_b1: tl.constexpr,
              block_b2: tl.constexpr,
              OUTPUT_LSE: tl.constexpr,
              HEAD_DIM: tl.constexpr,
              NUM_HEADS: tl.constexpr,
              BLOCK_I: tl.constexpr,
              BLOCK_K: tl.constexpr,
              BLOCK_J: tl.constexpr,
              BLOCK_HD: tl.constexpr,
              ):
    off_hfz = tl.program_id(0)
    off_z = off_hfz // (NUM_HEADS * nframes)
    off_qf = off_hfz // NUM_HEADS % nframes
    off_h = off_hfz % NUM_HEADS

    if isinstance(Lq_ptrs, tl.tensor_descriptor):
        dtype = Lq_ptrs.dtype
    else:
        dtype = Lq_ptrs.dtype.element_ty

    desc_lq = _maybe_make_tensor_desc(
        Lq_ptrs,
        shape=[bsz, nframes, block_b1, block_b2, NUM_HEADS * HEAD_DIM],
        strides=[nframes * block_b1 * block_b2 * NUM_HEADS * HEAD_DIM, block_b1 * block_b2 * NUM_HEADS * HEAD_DIM, block_b2 * NUM_HEADS * HEAD_DIM, NUM_HEADS * HEAD_DIM, 1],
        block_shape=[1, 1, BLOCK_I, BLOCK_J, HEAD_DIM]
    )
    desc_lk = _maybe_make_tensor_desc(
        Lk_ptrs,
        shape=[bsz, nframes, block_b1, NUM_HEADS, HEAD_DIM],
        strides=[nframes * block_b1 * NUM_HEADS * HEAD_DIM, block_b1 * NUM_HEADS * HEAD_DIM, NUM_HEADS * HEAD_DIM, HEAD_DIM, 1],
        block_shape=[1, 1, BLOCK_K, 1, BLOCK_HD]
    )
    desc_rq = _maybe_make_tensor_desc(
        Rq_ptrs,
        shape=[bsz, nframes, block_b2, NUM_HEADS, HEAD_DIM],
        strides=[nframes * block_b2 * NUM_HEADS * HEAD_DIM, block_b2 * NUM_HEADS * HEAD_DIM, NUM_HEADS * HEAD_DIM, HEAD_DIM, 1],
        block_shape=[1, 1, BLOCK_J, 1, HEAD_DIM]
    )
    desc_rk = _maybe_make_tensor_desc(
        Rk_ptrs,
        shape=[bsz, nframes, block_b1, block_b2, NUM_HEADS * HEAD_DIM],
        strides=[nframes * block_b1 * block_b2 * NUM_HEADS * HEAD_DIM, block_b1 * block_b2 * NUM_HEADS * HEAD_DIM, block_b2 * NUM_HEADS * HEAD_DIM, NUM_HEADS * HEAD_DIM, 1],
        block_shape=[1, 1, BLOCK_K, 1, BLOCK_HD]
    )
    desc_v = _maybe_make_tensor_desc(
        v_ptrs,
        shape=[bsz, nframes, block_b1, block_b2, NUM_HEADS * HEAD_DIM],
        strides=[nframes * block_b1 * block_b2 * NUM_HEADS * HEAD_DIM, block_b1 * block_b2 * NUM_HEADS * HEAD_DIM, block_b2 * NUM_HEADS * HEAD_DIM, NUM_HEADS * HEAD_DIM, 1],
        block_shape=[1, 1, BLOCK_K, 1, HEAD_DIM]
    )
    desc_o = _maybe_make_tensor_desc(
        o_ptrs,
        shape=[bsz, nframes, block_b1, block_b2, NUM_HEADS * HEAD_DIM],
        strides=[nframes * block_b1 * block_b2 * NUM_HEADS * HEAD_DIM, block_b1 * block_b2 * NUM_HEADS * HEAD_DIM, block_b2 * NUM_HEADS * HEAD_DIM, NUM_HEADS * HEAD_DIM, 1],
        block_shape=[1, 1, BLOCK_I, BLOCK_J, HEAD_DIM]
    )

    start_j = tl.program_id(1) * BLOCK_J
    start_i = tl.program_id(2) * BLOCK_I

    qk_scale = sm_scale * 1.44269504  # 1/log(2)

    l_ij = tl.zeros([BLOCK_I * BLOCK_J], dtype=tl.float32)
    m_ij = tl.full([BLOCK_I * BLOCK_J], dtype=tl.float32, value=float("-inf"))
    acc = tl.zeros([BLOCK_I * BLOCK_J, HEAD_DIM], dtype=tl.float32)

    tl.static_assert(HEAD_DIM % BLOCK_HD == 0, "HEAD_DIM must be divisible by BLOCK_HD")

    # i_mask = (start_i + tl.arange(0, BLOCK_I)) < block_b1
    # j_mask = (start_j + tl.arange(0, BLOCK_J)) < block_b2

    Lq_ij = desc_lq.load([off_z, off_qf, start_i, start_j, off_h * HEAD_DIM]).reshape(BLOCK_I * BLOCK_J, HEAD_DIM // BLOCK_HD, BLOCK_HD)
    Rq_j = desc_rq.load([off_z, off_qf, start_j, off_h, 0]).reshape(BLOCK_J, HEAD_DIM // BLOCK_HD, BLOCK_HD)
    # Rq_j = tl.where(j_mask[:, None, None], Rq_j, 0.0)

    for off_kf in tl.range(0, nframes):
        for k in tl.range(0, block_b1, BLOCK_K):
            k_mask = (k + tl.arange(0, BLOCK_K)) < block_b1
            L_jki = tl.zeros([BLOCK_I * BLOCK_J, BLOCK_K], dtype=tl.float32)
            for x in tl.static_range(0, HEAD_DIM, BLOCK_HD):
                Lk_k_x = desc_lk.load([off_z, off_kf, k, off_h, x]).reshape(BLOCK_K, BLOCK_HD)
                # Lk_k_x = tl.where(k_mask[:, None], Lk_k_x, 0.0)
                Lq_ij_x = tl.where(((tl.arange(0, HEAD_DIM // BLOCK_HD)) * BLOCK_HD)[None, :, None] == x, Lq_ij, 0.0).sum(1)
                L_jki += tl.dot(Lq_ij_x, Lk_k_x.T)
            L_jki = L_jki.reshape(BLOCK_I, BLOCK_J, BLOCK_K)
            L_jki = L_jki * qk_scale

            for l in tl.range(0, block_b2):
                R_kjl = tl.zeros([BLOCK_J, BLOCK_K], dtype=tl.float32)
                for x in tl.static_range(0, HEAD_DIM, BLOCK_HD):
                    Rk_kl_x = desc_rk.load([off_z, off_kf, k, l, off_h * HEAD_DIM + x]).reshape(BLOCK_K, BLOCK_HD)
                    # Rk_kl_x = tl.where(k_mask[:, None], Rk_kl_x, 0.0)
                    Rq_j_x = tl.where((tl.arange(0, HEAD_DIM // BLOCK_HD) * BLOCK_HD)[None, :, None] == x, Rq_j, 0.0).sum(1)
                    if BLOCK_J < 16:
                        R_kjl += (Rq_j_x.to(tl.float32).expand_dims(1) * Rk_kl_x.to(tl.float32).expand_dims(0)).sum(2)
                    else:
                        R_kjl += tl.dot(Rq_j_x, Rk_kl_x.T)
                R_kjl = R_kjl.reshape(1, BLOCK_J, BLOCK_K)

                # R_kjl = (Rq_j * Rk_kl).sum(1).reshape(1, BLOCK_K)
                # R_kjl = desc_r.load([off_hz, k, start_j, l]).reshape(1, BLOCK_L)
                v_kl = desc_v.load([off_z, off_kf, k, l, off_h * HEAD_DIM]).reshape(BLOCK_K, HEAD_DIM)
                v_kl = tl.where(k_mask[:, None], v_kl, 0.0)
                qk_ijkl = (L_jki * R_kjl).reshape(BLOCK_I * BLOCK_J, BLOCK_K)
                qk_ijkl = tl.where((k + tl.arange(0, BLOCK_K))[None, :] < block_b1, qk_ijkl, float("-inf"))
                m_ijk = tl.maximum(m_ij, tl.max(qk_ijkl, 1))
                qk_ijkl = qk_ijkl - m_ijk[:, None]
                p_ijkl = tl.exp2(qk_ijkl)

                alpha = tl.exp2(m_ij - m_ijk)
                l_ijk = tl.sum(p_ijkl, 1)
                acc = acc * alpha[:, None]

                p_ijkl = p_ijkl.to(dtype)
                acc = tl.dot(p_ijkl.reshape(BLOCK_I * BLOCK_J, BLOCK_K), v_kl, acc)

                l_ij = l_ij * alpha + l_ijk
                m_ij = m_ijk
    
    if OUTPUT_LSE:
        lse_ij = m_ij + tl.log2(l_ij)
        lse_ij = lse_ij.reshape(BLOCK_I, BLOCK_J)
        
        i_range = start_i + tl.arange(0, BLOCK_I)[:, None]
        j_range = start_j + tl.arange(0, BLOCK_J)[None, :]
        lse_ij_ptrs = lse_ptrs + (off_z * NUM_HEADS * nframes * block_b1 * block_b2
                                    + off_h * nframes * block_b1 * block_b2
                                    + off_qf * block_b1 * block_b2
                                    + i_range * block_b2
                                    + j_range)
        lse_ij_mask = (i_range < block_b1) & (j_range < block_b2)
        tl.store(lse_ij_ptrs, lse_ij, mask=lse_ij_mask)

    acc = acc / l_ij[:, None]
    acc = acc.to(dtype)
    desc_o.store([off_z, off_qf, start_i, start_j, off_h * HEAD_DIM], acc.reshape(1, 1, BLOCK_I, BLOCK_J, HEAD_DIM))

def _attn_bwd_preprocess_pre_hook(nargs):
    BLOCK_Q = nargs["BLOCK_Q"]
    BLOCK_HEAD_DIM = nargs["BLOCK_HEAD_DIM"]
    if not isinstance(nargs["o_ptr"], TensorDescriptor):
        return
    nargs["o_ptr"].block_shape = [1, BLOCK_Q, 1, BLOCK_HEAD_DIM]
    nargs["do_ptr"].block_shape = [1, BLOCK_Q, 1, BLOCK_HEAD_DIM]
    nargs["d_ptr"].block_shape = [1, 1, BLOCK_Q]

bwd_preprocess_configs = [
    triton.Config({'BLOCK_Q': BQ, 'BLOCK_HEAD_DIM': BHD}, num_stages=s, num_warps=w, pre_hook=_attn_bwd_preprocess_pre_hook) \
    for BQ in [16, 32, 64, 128]\
    for BHD in [16, 32, 64, 128]\
    for s in [2, 3, 4] \
    for w in [4, 8]\
]

bwd_preprocess_configs = [
    triton.Config({'BLOCK_Q': 32, 'BLOCK_HEAD_DIM': 64}, num_stages=4, num_warps=4, pre_hook=_attn_bwd_preprocess_pre_hook)
]

@triton.autotune(configs=bwd_preprocess_configs, key=["HEAD_DIM", "NUM_HEADS"])
@triton.jit
def _attn_bwd_preprocess(bsz, seq_len, o_ptr, do_ptr, d_ptr,
              HEAD_DIM: tl.constexpr,
              NUM_HEADS: tl.constexpr,
              BLOCK_Q: tl.constexpr,
              BLOCK_HEAD_DIM: tl.constexpr,
              ):
    
    off_hz = tl.program_id(0)
    off_z = off_hz // NUM_HEADS
    off_h = off_hz % NUM_HEADS

    off_q = tl.program_id(1) * BLOCK_Q

    if isinstance(o_ptr, tl.tensor_descriptor):
        dtype = o_ptr.dtype
    else:
        dtype = o_ptr.dtype.element_ty

    desc_o = _maybe_make_tensor_desc(
        o_ptr,
        shape=[bsz, seq_len, NUM_HEADS, HEAD_DIM],
        strides=[seq_len * NUM_HEADS * HEAD_DIM, NUM_HEADS * HEAD_DIM, HEAD_DIM, 1],
        block_shape=[1, BLOCK_Q, 1, BLOCK_HEAD_DIM]
    )
    desc_do = _maybe_make_tensor_desc(
        do_ptr,
        shape=[bsz, seq_len, NUM_HEADS, HEAD_DIM],
        strides=[seq_len * NUM_HEADS * HEAD_DIM, NUM_HEADS * HEAD_DIM, HEAD_DIM, 1],
        block_shape=[1, BLOCK_Q, 1, BLOCK_HEAD_DIM]
    )
    desc_d = _maybe_make_tensor_desc(
        d_ptr,
        shape=[bsz, NUM_HEADS, seq_len],
        strides=[NUM_HEADS * seq_len, seq_len, 1],
        block_shape=[1, 1, BLOCK_Q]
    )

    acc = tl.zeros([BLOCK_Q], dtype=dtype)
    tl.static_assert(HEAD_DIM % BLOCK_HEAD_DIM == 0, "HEAD_DIM must be multiple of BLOCK_HEAD_DIM")
    for i in tl.static_range(0, HEAD_DIM, BLOCK_HEAD_DIM):
        o_i = desc_o.load([off_z, off_q, off_h, i]).reshape(BLOCK_Q, BLOCK_HEAD_DIM)
        do_i = desc_do.load([off_z, off_q, off_h, i]).reshape(BLOCK_Q, BLOCK_HEAD_DIM)
        acc += (o_i * do_i).sum(1)
    desc_d.store([off_z, off_h, off_q], acc.reshape(1, 1, BLOCK_Q))


def _attn_bwd_pre_hook(nargs):
    BLOCK_I = nargs["BLOCK_I"]
    BLOCK_K = nargs["BLOCK_K"]
    BLOCK_J = nargs["BLOCK_J"]
    HEAD_DIM = nargs["HEAD_DIM"]
    if not isinstance(nargs["Lq_ptrs"], TensorDescriptor):
        return
    nargs["Lq_ptrs"].block_shape = [1, 1, BLOCK_I, BLOCK_J, HEAD_DIM]
    nargs["Lk_ptrs"].block_shape = [1, 1, BLOCK_K, 1, HEAD_DIM]
    nargs["Rq_ptrs"].block_shape = [1, 1, BLOCK_J, 1, HEAD_DIM]
    nargs["Rk_ptrs"].block_shape = [1, 1, BLOCK_K, 1, HEAD_DIM]
    nargs["v_ptrs"].block_shape = [1, 1, BLOCK_K, 1, HEAD_DIM]
    nargs["do_ptrs"].block_shape = [1, 1, BLOCK_I, BLOCK_J, HEAD_DIM]
    nargs["dLq_ptrs"].block_shape = [1, 1, BLOCK_I, BLOCK_J, HEAD_DIM]
    nargs["dRq_ptrs"].block_shape = [1, 1, BLOCK_J, 1, HEAD_DIM]
    nargs["dLk_ptrs"].block_shape = [1, 1, BLOCK_K, 1, HEAD_DIM]
    nargs["dRk_ptrs"].block_shape = [1, 1, BLOCK_K, 1, HEAD_DIM]
    nargs["dv_ptrs"].block_shape = [1, 1, BLOCK_K, 1, HEAD_DIM]
    # print(nargs["dv_ptrs"].strides, nargs["dv_ptrs"].block_shape, nargs["dv_ptrs"].shape)

attn_bwd_configs = [
    triton.Config({'BLOCK_I': BI, 'BLOCK_K': BK, 'BLOCK_J': BJ}, num_stages=s, num_warps=w, pre_hook=_attn_bwd_pre_hook) \
    for BI in [1, 2, 4, 16, 128]\
    for BK in [16, 32, 64]\
    for BJ in [1, 2, 4, 16, 32]\
    for s in [2, 3, 4] \
    for w in [4, 8]\
]

def my_filter(c):
    i = c.all_kwargs()["BLOCK_I"]
    k = c.all_kwargs()["BLOCK_K"]
    j = c.all_kwargs()["BLOCK_J"]
    return i * k >= 16 and i * j >= 16 and i * k * c.num_warps * c.num_stages < 32768

attn_bwd_configs = list(filter(my_filter, attn_bwd_configs))
print(len(attn_bwd_configs), "configs for attn_bwd")
# attn_bwd_configs = attn_bwd_configs[:1]

# 50% sparse:
# L40 config
# attn_bwd_configs = [
#     triton.Config({'BLOCK_I': 1, 'BLOCK_K': 16, 'BLOCK_J': 32}, num_stages=2, num_warps=4, pre_hook=_attn_bwd_pre_hook)
# ]
# H100 config
# attn_bwd_configs = [
#     triton.Config({'BLOCK_I': 2, 'BLOCK_K': 16, 'BLOCK_J': 16}, num_stages=3, num_warps=4, pre_hook=_attn_bwd_pre_hook)
# ]

# 95% sparse:
# L40 config
# attn_bwd_configs = [
#     triton.Config({'BLOCK_I': 2, 'BLOCK_K': 32, 'BLOCK_J': 16}, num_stages=2, num_warps=4, pre_hook=_attn_bwd_pre_hook)
# ]
# H100 config
attn_bwd_configs = [
    triton.Config({'BLOCK_I': 1, 'BLOCK_K': 32, 'BLOCK_J': 16}, num_stages=3, num_warps=4, pre_hook=_attn_bwd_pre_hook)
]

@triton.autotune(configs=attn_bwd_configs, key=["HEAD_DIM", "NUM_HEADS", "block_b1", "block_b2"])
@triton.jit
def _attn_bwd(sm_scale, bsz,
              nframes,
              Lq_ptrs, Lk_ptrs,
              Rq_ptrs, Rk_ptrs,
              v_ptrs, do_ptrs,
              d_ptrs, lse_ptrs,
              dLq_ptrs, dLk_ptrs,
              dRq_ptrs, dRk_ptrs,
              dv_ptrs,
              block_b1: tl.constexpr,
              block_b2: tl.constexpr,
              HEAD_DIM: tl.constexpr,
              NUM_HEADS: tl.constexpr,
              BLOCK_I: tl.constexpr,
              BLOCK_K: tl.constexpr,
              BLOCK_J: tl.constexpr,
              ):
    off_hfz = tl.program_id(0)
    off_z = off_hfz // (NUM_HEADS * nframes * nframes)
    off_qf = (off_hfz // (NUM_HEADS * nframes)) % nframes
    off_kf = (off_hfz // NUM_HEADS) % nframes
    off_h = off_hfz % NUM_HEADS

    if isinstance(Lq_ptrs, tl.tensor_descriptor):
        dtype = Lq_ptrs.dtype
        out_dtype = dLq_ptrs.dtype
    else:
        dtype = Lq_ptrs.dtype.element_ty
        out_dtype = dLq_ptrs.dtype.element_ty

    desc_lq = _maybe_make_tensor_desc(
        Lq_ptrs,
        shape=[bsz, nframes, block_b1, block_b2, NUM_HEADS * HEAD_DIM],
        strides=[nframes * block_b1 * block_b2 * NUM_HEADS * HEAD_DIM, block_b1 * block_b2 * NUM_HEADS * HEAD_DIM, block_b2 * NUM_HEADS * HEAD_DIM, NUM_HEADS * HEAD_DIM, 1],
        block_shape=[1, 1, BLOCK_I, BLOCK_J, HEAD_DIM]
    )
    desc_lk = _maybe_make_tensor_desc(
        Lk_ptrs,
        shape=[bsz, nframes, block_b1, NUM_HEADS, HEAD_DIM],
        strides=[nframes * block_b1 * NUM_HEADS * HEAD_DIM, block_b1 * NUM_HEADS * HEAD_DIM, NUM_HEADS * HEAD_DIM, HEAD_DIM, 1],
        block_shape=[1, 1, BLOCK_K, 1, HEAD_DIM]
    )
    desc_rq = _maybe_make_tensor_desc(
        Rq_ptrs,
        shape=[bsz, nframes, block_b2, NUM_HEADS, HEAD_DIM],
        strides=[nframes * block_b2 * NUM_HEADS * HEAD_DIM, block_b2 * NUM_HEADS * HEAD_DIM, NUM_HEADS * HEAD_DIM, HEAD_DIM, 1],
        block_shape=[1, 1, BLOCK_J, 1, HEAD_DIM]
    )
    desc_rk = _maybe_make_tensor_desc(
        Rk_ptrs,
        shape=[bsz, nframes, block_b1, block_b2, NUM_HEADS * HEAD_DIM],
        strides=[nframes * block_b1 * block_b2 * NUM_HEADS * HEAD_DIM, block_b1 * block_b2 * NUM_HEADS * HEAD_DIM, block_b2 * NUM_HEADS * HEAD_DIM, NUM_HEADS * HEAD_DIM, 1],
        block_shape=[1, 1, BLOCK_K, 1, HEAD_DIM]
    )
    desc_v = _maybe_make_tensor_desc(
        v_ptrs,
        shape=[bsz, nframes, block_b1, block_b2, NUM_HEADS * HEAD_DIM],
        strides=[nframes * block_b1 * block_b2 * NUM_HEADS * HEAD_DIM, block_b1 * block_b2 * NUM_HEADS * HEAD_DIM, block_b2 * NUM_HEADS * HEAD_DIM, NUM_HEADS * HEAD_DIM, 1],
        block_shape=[1, 1, BLOCK_K, 1, HEAD_DIM]
    )
    desc_do = _maybe_make_tensor_desc(
        do_ptrs,
        shape=[bsz, nframes, block_b1, block_b2, NUM_HEADS * HEAD_DIM],
        strides=[nframes * block_b1 * block_b2 * NUM_HEADS * HEAD_DIM, block_b1 * block_b2 * NUM_HEADS * HEAD_DIM, block_b2 * NUM_HEADS * HEAD_DIM, NUM_HEADS * HEAD_DIM, 1],
        block_shape=[1, 1, BLOCK_I, BLOCK_J, HEAD_DIM]
    )

    desc_dlq = _maybe_make_tensor_desc(
        dLq_ptrs,
        shape=[bsz, nframes, block_b1, block_b2, NUM_HEADS * HEAD_DIM],
        strides=[nframes * block_b1 * block_b2 * NUM_HEADS * HEAD_DIM, block_b1 * block_b2 * NUM_HEADS * HEAD_DIM, block_b2 * NUM_HEADS * HEAD_DIM, NUM_HEADS * HEAD_DIM, 1],
        block_shape=[1, 1, BLOCK_I, BLOCK_J, HEAD_DIM]
    )
    desc_drq = _maybe_make_tensor_desc(
        dRq_ptrs,
        shape=[bsz, nframes, block_b2, NUM_HEADS, HEAD_DIM],
        strides=[nframes * block_b2 * NUM_HEADS * HEAD_DIM, block_b2 * NUM_HEADS * HEAD_DIM, NUM_HEADS * HEAD_DIM, HEAD_DIM, 1],
        block_shape=[1, 1, BLOCK_J, 1, HEAD_DIM]
    )
    desc_dlk = _maybe_make_tensor_desc(
        dLk_ptrs,
        shape=[bsz, nframes, block_b1, NUM_HEADS, HEAD_DIM],
        strides=[nframes * block_b1 * NUM_HEADS * HEAD_DIM, block_b1 * NUM_HEADS * HEAD_DIM, NUM_HEADS * HEAD_DIM, HEAD_DIM, 1],
        block_shape=[1, 1, BLOCK_K, 1, HEAD_DIM]
    )
    desc_drk = _maybe_make_tensor_desc(
        dRk_ptrs,
        shape=[bsz, nframes, block_b1, block_b2, NUM_HEADS * HEAD_DIM],
        strides=[nframes * block_b1 * block_b2 * NUM_HEADS * HEAD_DIM, block_b1 * block_b2 * NUM_HEADS * HEAD_DIM, block_b2 * NUM_HEADS * HEAD_DIM, NUM_HEADS * HEAD_DIM, 1],
        block_shape=[1, 1, BLOCK_K, 1, HEAD_DIM]
    )
    desc_dv = _maybe_make_tensor_desc(
        dv_ptrs,
        shape=[bsz, nframes, block_b1, block_b2, NUM_HEADS * HEAD_DIM],
        strides=[nframes * block_b1 * block_b2 * NUM_HEADS * HEAD_DIM, block_b1 * block_b2 * NUM_HEADS * HEAD_DIM, block_b2 * NUM_HEADS * HEAD_DIM, NUM_HEADS * HEAD_DIM, 1],
        block_shape=[1, 1, BLOCK_K, 1, HEAD_DIM]
    )

    start_j = tl.program_id(1) * BLOCK_J
    start_k = tl.program_id(2) * BLOCK_K

    qk_scale = sm_scale * 1.44269504  # 1/log(2)

    k_mask = (start_k + tl.arange(0, BLOCK_K)) < block_b1

    Lk_k = desc_lk.load([off_z, off_kf, start_k, off_h, 0]).reshape(BLOCK_K, HEAD_DIM)
    Rq_j = desc_rq.load([off_z, off_qf, start_j, off_h, 0]).reshape(BLOCK_J, HEAD_DIM)

    dlk_k = tl.zeros([BLOCK_K, HEAD_DIM], dtype=tl.float32)
    drq_j = tl.zeros([BLOCK_J, HEAD_DIM], dtype=tl.float32)

    j_range = start_j + tl.arange(0, BLOCK_J)[None, :]
    j_mask = j_range < block_b2
    
    off_d_lse = off_z * NUM_HEADS * nframes * block_b1 * block_b2 + off_h * nframes * block_b1 * block_b2 + off_qf * block_b1 * block_b2 + j_range
    d_ptrs = d_ptrs + off_d_lse
    lse_ptrs = lse_ptrs + off_d_lse

    for l in tl.range(0, block_b2):
        Rk_kl = desc_rk.load([off_z, off_kf, start_k, l, off_h * HEAD_DIM]).reshape(BLOCK_K, HEAD_DIM)
        v_kl = desc_v.load([off_z, off_kf, start_k, l, off_h * HEAD_DIM]).reshape(BLOCK_K, HEAD_DIM)
        if BLOCK_J >= 16:
            R_kjl = tl.dot(Rq_j, Rk_kl.T).expand_dims(0) # (1, BLOCK_J, BLOCK_K)
        else:
            R_kjl = (Rq_j.to(tl.float32).expand_dims(1) * Rk_kl.to(tl.float32).expand_dims(0)).sum(2).expand_dims(0)

        dv_jkl = tl.zeros([BLOCK_K, HEAD_DIM], dtype=tl.float32)
        dr_jkl = tl.zeros([BLOCK_J, BLOCK_K], dtype=tl.float32)

        for i in tl.range(0, block_b1, BLOCK_I):
            i_range = i + tl.arange(0, BLOCK_I)
            i_mask = i_range < block_b1

            Lq_ij = desc_lq.load([off_z, off_qf, i, start_j, off_h * HEAD_DIM]).reshape(BLOCK_I * BLOCK_J, HEAD_DIM)
            L_jki = tl.dot(Lq_ij, Lk_k.T).reshape(BLOCK_I, BLOCK_J, BLOCK_K) # (BLOCK_I * BLOCK_J, BLOCK_K)
            ij_mask = i_mask[:, None] & j_mask
            lse_ij = tl.load(lse_ptrs + (i_range * block_b2)[:, None], mask=ij_mask)

            d_ij = tl.load(d_ptrs + (i_range * block_b2)[:, None], mask=ij_mask)
            do_ij = desc_do.load([off_z, off_qf, i, start_j, off_h * HEAD_DIM]).reshape(BLOCK_I, BLOCK_J, HEAD_DIM)

            s_ijkl = (L_jki * R_kjl) * qk_scale # (BLOCK_I, BLOCK_J, BLOCK_K)

            p_ijkl = tl.exp2(s_ijkl - lse_ij[:, :, None]).to(dtype)
            p_ijkl = tl.where(ij_mask[:, :, None], p_ijkl, 0.0)
            do_ij = tl.where(ij_mask[:, :, None], do_ij, 0.0).reshape(BLOCK_I * BLOCK_J, HEAD_DIM)
            dv_jkl += tl.dot(p_ijkl.reshape(BLOCK_I * BLOCK_J, BLOCK_K).T, do_ij).to(out_dtype) # (BLOCK_K, HEAD_DIM)

            dp_ijkl = tl.dot(do_ij, v_kl.T).reshape(BLOCK_I, BLOCK_J, BLOCK_K)
            ds_ijkl = (p_ijkl * (dp_ijkl - d_ij[:, :, None]) * sm_scale)
            ds_ijkl = tl.where(ij_mask[:, :, None] & k_mask[None, None, :], ds_ijkl, 0.0)

            dl_jik_l = (ds_ijkl * R_kjl).reshape(BLOCK_I * BLOCK_J, BLOCK_K).to(dtype)
            dlq_ij_l = tl.dot(dl_jik_l, Lk_k).to(out_dtype) # (BLOCK_I * BLOCK_J, HEAD_DIM)
            desc_dlq.atomic_add([off_z, off_qf, i, start_j, off_h * HEAD_DIM], dlq_ij_l.reshape(1, 1, BLOCK_I, BLOCK_J, HEAD_DIM))

            dlk_k += tl.dot(dl_jik_l.T, Lq_ij) # (BLOCK_K, HEAD_DIM)
            dr_jkl += tl.where(ij_mask[:, :, None], ds_ijkl * L_jki, 0.0).sum(0) # (BLOCK_J, BLOCK_K)
        
        dr_jkl = tl.where(k_mask[None, :], dr_jkl, 0.0)
        if BLOCK_J >= 16:
            dr_jkl = dr_jkl.to(dtype)
            drq_j += tl.dot(dr_jkl, Rk_kl) # (BLOCK_J, HEAD_DIM)
            drk_jkl = tl.dot(dr_jkl.T, Rq_j).to(out_dtype) # (BLOCK_K, HEAD_DIM)
        else:
            drq_j += (dr_jkl.expand_dims(2) * Rk_kl.expand_dims(0)).sum(1)
            drk_jkl = (dr_jkl.expand_dims(2) * Rq_j.expand_dims(1)).sum(0).to(out_dtype)
        desc_drk.atomic_add([off_z, off_kf, start_k, l, off_h * HEAD_DIM], drk_jkl.reshape(1, 1, BLOCK_K, 1, HEAD_DIM))
        desc_dv.atomic_add([off_z, off_kf, start_k, l, off_h * HEAD_DIM], dv_jkl.to(out_dtype).reshape(1, 1, BLOCK_K, 1, HEAD_DIM))

    desc_drq.atomic_add([off_z, off_qf, start_j, off_h, 0], drq_j.to(out_dtype).reshape(1, 1, BLOCK_J, 1, HEAD_DIM))
    desc_dlk.atomic_add([off_z, off_kf, start_k, off_h, 0], dlk_k.to(out_dtype).reshape(1, 1, BLOCK_K, 1, HEAD_DIM))

class MonarchAttention(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda')
    def forward(ctx, Lq, Lk, Rq, Rk, v, sm_scale, grad_enabled):
        bsz, nframes, block_b1, block_b2, nh, head_dim = Lq.size()
        seq_len = nframes * block_b1 * block_b2
        assert Lk.size() == (bsz, nframes, block_b1, nh, head_dim)
        assert Rq.size() == (bsz, nframes, block_b2, nh, head_dim)
        assert Rk.size() == (bsz, nframes, block_b1, block_b2, nh, head_dim)
        assert v.size() == (bsz, seq_len, nh, head_dim)

        grad_on = grad_enabled and any(x.requires_grad for x in [Lq, Lk, Rq, Rk, v])

        o = torch.empty((bsz, seq_len, nh, head_dim), device=Lq.device, dtype=Lq.dtype)
        lse_vals = torch.empty((bsz, nh, nframes, block_b1, block_b2), device=Lq.device, dtype=torch.float32) if grad_on else None

        if supports_host_descriptor:
            desc_lq = TensorDescriptor(Lq, shape=[bsz, nframes, block_b1, block_b2, nh * head_dim],
                                    strides=[nframes * block_b1 * block_b2 * nh * head_dim, block_b1 * block_b2 * nh * head_dim, block_b2 * nh * head_dim, nh * head_dim, 1],
                                    block_shape=[1, 1, 1, 1, 1])
            desc_lk = TensorDescriptor(Lk, shape=[bsz, nframes, block_b1, nh, head_dim],
                                    strides=[nframes * block_b1 * nh * head_dim, block_b1 * nh * head_dim, nh * head_dim, head_dim, 1],
                                    block_shape=[1, 1, 1, 1, 1])
            desc_rq = TensorDescriptor(Rq, shape=[bsz, nframes, block_b2, nh, head_dim],
                                    strides=[nframes * block_b2 * nh * head_dim, block_b2 * nh * head_dim, nh * head_dim, head_dim, 1],
                                    block_shape=[1, 1, 1, 1, 1])
            desc_rk = TensorDescriptor(Rk, shape=[bsz, nframes, block_b1, block_b2, nh * head_dim],
                                    strides=[nframes * block_b1 * block_b2 * nh * head_dim, block_b1 * block_b2 * nh * head_dim, block_b2 * nh * head_dim, nh * head_dim, 1],
                                    block_shape=[1, 1, 1, 1, 1])
            desc_v = TensorDescriptor(v, shape=[bsz, nframes, block_b1, block_b2, nh * head_dim],
                                    strides=[nframes * block_b1 * block_b2 * nh * head_dim, block_b1 * block_b2 * nh * head_dim, block_b2 * nh * head_dim, nh * head_dim, 1],
                                    block_shape=[1, 1, 1, 1, 1])
            desc_o = TensorDescriptor(o, shape=[bsz, nframes, block_b1, block_b2, nh * head_dim],
                                    strides=[nframes * block_b1 * block_b2 * nh * head_dim, block_b1 * block_b2 * nh * head_dim, block_b2 * nh * head_dim, nh * head_dim, 1],
                                    block_shape=[1, 1, 1, 1, 1])
        else:
            desc_lq = Lq
            desc_lk = Lk
            desc_rq = Rq
            desc_rk = Rk
            desc_v = v
            desc_o = o

        def alloc_fn(size: int, align: int, _):
            return torch.empty(size, dtype=torch.int8, device="cuda")
        triton.set_allocator(alloc_fn)

        def grid(META):
            return (bsz * nh * nframes, triton.cdiv(block_b2, META["BLOCK_J"]), triton.cdiv(block_b1, META["BLOCK_I"]))

        _attn_fwd[grid](
            sm_scale, bsz,
            nframes,
            desc_lq, desc_lk,
            desc_rq, desc_rk,
            desc_v, desc_o,
            lse_vals,
            block_b1, block_b2,
            grad_on,
            head_dim, nh
        )

        if grad_on:
            ctx.save_for_backward(Lq, Lk, Rq, Rk, v, o, lse_vals)
            ctx.sm_scale = sm_scale

        return o

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(ctx, dout):
        Lq, Lk, Rq, Rk, v, o, lse = ctx.saved_tensors
        sm_scale = ctx.sm_scale

        bsz, nframes, block_b1, block_b2, nh, head_dim = Lq.size()
        seq_len = nframes * block_b1 * block_b2

        dout = dout.contiguous()

        D = torch.empty((bsz, nh, seq_len), device=o.device, dtype=o.dtype)
        if supports_host_descriptor:
            desc_o = TensorDescriptor(o, shape=[bsz, seq_len, nh, head_dim],
                                    strides=[seq_len * nh * head_dim, nh * head_dim, head_dim, 1],
                                    block_shape=[1, 1, 1, 1])
            desc_do = TensorDescriptor(dout, shape=[bsz, seq_len, nh, head_dim],
                                    strides=[seq_len * nh * head_dim, nh * head_dim, head_dim, 1],
                                    block_shape=[1, 1, 1, 1])
            desc_d = TensorDescriptor(D, shape=[bsz, nh, seq_len],
                                    strides=[nh * seq_len, seq_len, 1],
                                    block_shape=[1, 1, 1])
        else:
            desc_o = o
            desc_do = dout
            desc_d = D

        def alloc_fn(size: int, align: int, _):
            return torch.empty(size, dtype=torch.int8, device="cuda")
        triton.set_allocator(alloc_fn)

        def grid(META):
            return (bsz * nh, triton.cdiv(seq_len, META["BLOCK_Q"]))

        _attn_bwd_preprocess[grid](
            bsz, seq_len, desc_o, desc_do, desc_d,
            head_dim, nh
        )

        dLq = torch.zeros_like(Lq)
        dRq = torch.zeros_like(Rq)
        dLk = torch.zeros_like(Lk)
        dRk = torch.zeros_like(Rk)
        dv = torch.zeros_like(v)

        if supports_host_descriptor:
            desc_lq = TensorDescriptor(Lq, shape=[bsz, nframes, block_b1, block_b2, nh * head_dim],
                                    strides=[nframes * block_b1 * block_b2 * nh * head_dim, block_b1 * block_b2 * nh * head_dim, block_b2 * nh * head_dim, nh * head_dim, 1],
                                    block_shape=[1, 1, 1, 1, 1])
            desc_lk = TensorDescriptor(Lk, shape=[bsz, nframes, block_b1, nh, head_dim],
                                    strides=[nframes * block_b1 * nh * head_dim, block_b1 * nh * head_dim, nh * head_dim, head_dim, 1],
                                    block_shape=[1, 1, 1, 1, 1])
            desc_rq = TensorDescriptor(Rq, shape=[bsz, nframes, block_b2, nh, head_dim],
                                    strides=[nframes * block_b2 * nh * head_dim, block_b2 * nh * head_dim, nh * head_dim, head_dim, 1],
                                    block_shape=[1, 1, 1, 1, 1])
            desc_rk = TensorDescriptor(Rk, shape=[bsz, nframes, block_b1, block_b2, nh * head_dim],
                                    strides=[nframes * block_b1 * block_b2 * nh * head_dim, block_b1 * block_b2 * nh * head_dim, block_b2 * nh * head_dim, nh * head_dim, 1],
                                    block_shape=[1, 1, 1, 1, 1])
            desc_v = TensorDescriptor(v, shape=[bsz, nframes, block_b1, block_b2, nh * head_dim],
                                    strides=[nframes * block_b1 * block_b2 * nh * head_dim, block_b1 * block_b2 * nh * head_dim, block_b2 * nh * head_dim, nh * head_dim, 1],
                                    block_shape=[1, 1, 1, 1, 1])
            desc_do = TensorDescriptor(dout, shape=[bsz, nframes, block_b1, block_b2, nh * head_dim],
                                    strides=[nframes * block_b1 * block_b2 * nh * head_dim, block_b1 * block_b2 * nh * head_dim, block_b2 * nh * head_dim, nh * head_dim, 1],
                                    block_shape=[1, 1, 1, 1, 1])
            desc_dlq = TensorDescriptor(dLq, shape=[bsz, nframes, block_b1, block_b2, nh * head_dim],
                                    strides=[nframes * block_b1 * block_b2 * nh * head_dim, block_b1 * block_b2 * nh * head_dim, block_b2 * nh * head_dim, nh * head_dim, 1],
                                    block_shape=[1, 1, 1, 1, 1])
            desc_drq = TensorDescriptor(dRq, shape=[bsz, nframes, block_b2, nh, head_dim],
                                    strides=[nframes * block_b2 * nh * head_dim, block_b2 * nh * head_dim, nh * head_dim, head_dim, 1],
                                    block_shape=[1, 1, 1, 1, 1])
            desc_dlk = TensorDescriptor(dLk, shape=[bsz, nframes, block_b1, nh, head_dim],
                                    strides=[nframes * block_b1 * nh * head_dim, block_b1 * nh * head_dim, nh * head_dim, head_dim, 1],
                                    block_shape=[1, 1, 1, 1, 1])
            desc_drk = TensorDescriptor(dRk, shape=[bsz, nframes, block_b1, block_b2, nh * head_dim],
                                    strides=[nframes * block_b1 * block_b2 * nh * head_dim, block_b1 * block_b2 * nh * head_dim, block_b2 * nh * head_dim, nh * head_dim, 1],
                                    block_shape=[1, 1, 1, 1, 1])
            desc_dv = TensorDescriptor(dv, shape=[bsz, nframes, block_b1, block_b2, nh * head_dim],
                                    strides=[nframes * block_b1 * block_b2 * nh * head_dim, block_b1 * block_b2 * nh * head_dim, block_b2 * nh * head_dim, nh * head_dim, 1],
                                    block_shape=[1, 1, 1, 1, 1])
        else:
            desc_lq = Lq
            desc_lk = Lk
            desc_rq = Rq
            desc_rk = Rk
            desc_v = v
            desc_do = dout
            desc_dlq = dLq
            desc_drq = dRq
            desc_dlk = dLk
            desc_drk = dRk
            desc_dv = dv
        
        # print(dv.view(bsz, block_b1, block_b2, nh, head_dim).stride(), [block_b1 * block_b2 * nh * head_dim, block_b2 * nh * head_dim, nh * head_dim, head_dim, 1])

        def grid(META):
            return (bsz * nh * nframes * nframes, triton.cdiv(block_b2, META["BLOCK_J"]), triton.cdiv(block_b1, META["BLOCK_K"]))

        _attn_bwd[grid](
            sm_scale, bsz, nframes,
            desc_lq, desc_lk,
            desc_rq, desc_rk,
            desc_v, desc_do,
            D, lse,
            desc_dlq, desc_dlk,
            desc_drq, desc_drk,
            desc_dv,
            block_b1, block_b2,
            head_dim, nh
        )

        return dLq.to(Lq.dtype), dLk.to(Lk.dtype), dRq.to(Rq.dtype), dRk.to(Rk.dtype), dv.to(v.dtype), None, None, None, None, None, None

monarch_attn = MonarchAttention.apply
