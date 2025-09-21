# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
from einops import rearrange

from fastvideo.attention.selector import backend_name_to_enum, get_attn_backend
from fastvideo.distributed.communication_op import (
    sequence_model_parallel_all_gather, sequence_model_parallel_all_to_all_4D)
from fastvideo.distributed.parallel_state import (get_sp_parallel_rank,
                                                  get_sp_world_size)
from fastvideo.forward_context import ForwardContext, get_forward_context
from fastvideo.platforms import AttentionBackendEnum
from fastvideo.utils import get_compute_dtype
from fastvideo.layers.rotary_embedding import _apply_rotary_emb
from fastvideo.layers.linear import BatchedReplicatedLinear
from fastvideo.attention.flash_attn import flash_attention
import fastvideo.envs as envs
import math

class DistributedAttention(nn.Module):
    """Distributed attention layer.
    """

    def __init__(self,
                 num_heads: int,
                 head_size: int,
                 num_kv_heads: int | None = None,
                 softmax_scale: float | None = None,
                 causal: bool = False,
                 supported_attention_backends: tuple[AttentionBackendEnum, ...]
                 | None = None,
                 prefix: str = "",
                 **extra_impl_args) -> None:
        super().__init__()
        if softmax_scale is None:
            self.softmax_scale = head_size**-0.5
        else:
            self.softmax_scale = softmax_scale

        if num_kv_heads is None:
            num_kv_heads = num_heads

        dtype = get_compute_dtype()
        attn_backend = get_attn_backend(
            head_size,
            dtype,
            supported_attention_backends=supported_attention_backends)
        impl_cls = attn_backend.get_impl_cls()
        self.attn_impl = impl_cls(num_heads=num_heads,
                                  head_size=head_size,
                                  causal=causal,
                                  softmax_scale=self.softmax_scale,
                                  num_kv_heads=num_kv_heads,
                                  prefix=f"{prefix}.impl",
                                  **extra_impl_args)
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads
        self.backend = backend_name_to_enum(attn_backend.get_name())
        self.dtype = dtype

    @torch.compiler.disable
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        replicated_q: torch.Tensor | None = None,
        replicated_k: torch.Tensor | None = None,
        replicated_v: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass for distributed attention.
        
        Args:
            q (torch.Tensor): Query tensor [batch_size, seq_len, num_heads, head_dim]
            k (torch.Tensor): Key tensor [batch_size, seq_len, num_heads, head_dim]
            v (torch.Tensor): Value tensor [batch_size, seq_len, num_heads, head_dim]
            replicated_q (Optional[torch.Tensor]): Replicated query tensor, typically for text tokens
            replicated_k (Optional[torch.Tensor]): Replicated key tensor
            replicated_v (Optional[torch.Tensor]): Replicated value tensor
            
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: A tuple containing:
                - o (torch.Tensor): Output tensor after attention for the main sequence
                - replicated_o (Optional[torch.Tensor]): Output tensor for replicated tokens, if provided
        """
        # Check input shapes
        assert q.dim() == 4 and k.dim() == 4 and v.dim(
        ) == 4, "Expected 4D tensors"
        batch_size, seq_len, num_heads, head_dim = q.shape
        local_rank = get_sp_parallel_rank()
        world_size = get_sp_world_size()

        forward_context: ForwardContext = get_forward_context()
        ctx_attn_metadata = forward_context.attn_metadata

        # Stack QKV
        qkv = torch.cat([q, k, v], dim=0)  # [3, seq_len, num_heads, head_dim]

        # Redistribute heads across sequence dimension
        qkv = sequence_model_parallel_all_to_all_4D(qkv,
                                                    scatter_dim=2,
                                                    gather_dim=1)
        # Apply backend-specific preprocess_qkv
        qkv = self.attn_impl.preprocess_qkv(qkv, ctx_attn_metadata)

        # Concatenate with replicated QKV if provided
        if replicated_q is not None:
            assert replicated_k is not None and replicated_v is not None
            replicated_qkv = torch.cat(
                [replicated_q, replicated_k, replicated_v],
                dim=0)  # [3, seq_len, num_heads, head_dim]
            heads_per_rank = num_heads // world_size
            replicated_qkv = replicated_qkv[:, :, local_rank *
                                            heads_per_rank:(local_rank + 1) *
                                            heads_per_rank]
            qkv = torch.cat([qkv, replicated_qkv], dim=1)

        q, k, v = qkv.chunk(3, dim=0)

        output = self.attn_impl.forward(q, k, v, ctx_attn_metadata)

        # Redistribute back if using sequence parallelism
        replicated_output = None
        if replicated_q is not None:
            replicated_output = output[:, seq_len * world_size:]
            output = output[:, :seq_len * world_size]
            # TODO: make this asynchronous
            replicated_output = sequence_model_parallel_all_gather(
                replicated_output.contiguous(), dim=2)
        # Apply backend-specific postprocess_output
        output = self.attn_impl.postprocess_output(output, ctx_attn_metadata)

        output = sequence_model_parallel_all_to_all_4D(output,
                                                       scatter_dim=1,
                                                       gather_dim=2)
        return output, replicated_output


class DistributedAttention_VSA(DistributedAttention):
    """Distributed attention layer with VSA support.
    """

    @torch.compiler.disable
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        replicated_q: torch.Tensor | None = None,
        replicated_k: torch.Tensor | None = None,
        replicated_v: torch.Tensor | None = None,
        gate_compress: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass for distributed attention.
        
        Args:
            q (torch.Tensor): Query tensor [batch_size, seq_len, num_heads, head_dim]
            k (torch.Tensor): Key tensor [batch_size, seq_len, num_heads, head_dim]
            v (torch.Tensor): Value tensor [batch_size, seq_len, num_heads, head_dim]
            gate_compress (torch.Tensor): Gate compress tensor [batch_size, seq_len, num_heads, head_dim]
            replicated_q (Optional[torch.Tensor]): Replicated query tensor, typically for text tokens
            replicated_k (Optional[torch.Tensor]): Replicated key tensor
            replicated_v (Optional[torch.Tensor]): Replicated value tensor
            
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: A tuple containing:
                - o (torch.Tensor): Output tensor after attention for the main sequence
                - replicated_o (Optional[torch.Tensor]): Output tensor for replicated tokens, if provided
        """
        # Check text tokens are not supported for VSA now
        assert replicated_q is None and replicated_k is None and replicated_v is None, "Replicated QKV is not supported for VSA now"
        # Check input shapes
        assert q.dim() == 4 and k.dim() == 4 and v.dim(
        ) == 4, "Expected 4D tensors"

        forward_context: ForwardContext = get_forward_context()
        ctx_attn_metadata = forward_context.attn_metadata

        # Stack QKV
        qkvg = torch.cat([q, k, v, gate_compress],
                         dim=0)  # [3, seq_len, num_heads, head_dim]

        # Redistribute heads across sequence dimension
        qkvg = sequence_model_parallel_all_to_all_4D(qkvg,
                                                     scatter_dim=2,
                                                     gather_dim=1)

        qkvg = self.attn_impl.preprocess_qkv(qkvg, ctx_attn_metadata)

        q, k, v, gate_compress = qkvg.chunk(4, dim=0)
        output = self.attn_impl.forward(
            q, k, v, gate_compress, ctx_attn_metadata)  # type: ignore[call-arg]

        # Redistribute back if using sequence parallelism
        replicated_output = None

        # Apply backend-specific postprocess_output
        output = self.attn_impl.postprocess_output(output, ctx_attn_metadata)

        output = sequence_model_parallel_all_to_all_4D(output,
                                                       scatter_dim=1,
                                                       gather_dim=2)
        return output, replicated_output


class LocalAttention(nn.Module):
    """Attention layer.
    """

    def __init__(self,
                 num_heads: int,
                 head_size: int,
                 num_kv_heads: int | None = None,
                 softmax_scale: float | None = None,
                 causal: bool = False,
                 supported_attention_backends: tuple[AttentionBackendEnum, ...]
                 | None = None,
                 **extra_impl_args) -> None:
        super().__init__()
        if softmax_scale is None:
            self.softmax_scale = head_size**-0.5
        else:
            self.softmax_scale = softmax_scale
        if num_kv_heads is None:
            num_kv_heads = num_heads

        dtype = get_compute_dtype()
        attn_backend = get_attn_backend(
            head_size,
            dtype,
            supported_attention_backends=supported_attention_backends)
        impl_cls = attn_backend.get_impl_cls()
        self.attn_impl = impl_cls(num_heads=num_heads,
                                  head_size=head_size,
                                  softmax_scale=self.softmax_scale,
                                  num_kv_heads=num_kv_heads,
                                  causal=causal,
                                  **extra_impl_args)
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads
        self.backend = backend_name_to_enum(attn_backend.get_name())
        self.dtype = dtype

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply local attention between query, key and value tensors.
        
        Args:
            q (torch.Tensor): Query tensor of shape [batch_size, seq_len, num_heads, head_dim]
            k (torch.Tensor): Key tensor of shape [batch_size, seq_len, num_heads, head_dim] 
            v (torch.Tensor): Value tensor of shape [batch_size, seq_len, num_heads, head_dim]
            
        Returns:
            torch.Tensor: Output tensor after local attention
        """
        # Check input shapes
        assert q.dim() == 4 and k.dim() == 4 and v.dim(
        ) == 4, "Expected 4D tensors"

        forward_context: ForwardContext = get_forward_context()
        ctx_attn_metadata = forward_context.attn_metadata

        output = self.attn_impl.forward(q, k, v, ctx_attn_metadata)
        return output

class MonarchAttention(nn.Module):

    def __init__(self,
                 num_heads: int,
                 head_size: int,
                 softmax_scale: float | None = None,
                 causal: bool = False,
                 prefix: str = "",
                 **extra_impl_args) -> None:
        super().__init__()
        if softmax_scale is None:
            self.softmax_scale = head_size**-0.5
        else:
            self.softmax_scale = softmax_scale

        assert not causal, "assuming non-causal for now"

        self.use_dynamic = envs.FASTVIDEO_MONARCH_USE_DYNAMIC

        dtype = get_compute_dtype()
        self.num_heads = num_heads
        self.head_size = head_size
        self.dtype = dtype

        self.to_lkq = BatchedReplicatedLinear(self.head_size, self.head_size, num_heads, bias=True, prefix=f"{prefix}.to_lkq")
        self.to_lkk = BatchedReplicatedLinear(self.head_size, self.head_size, num_heads, bias=True, prefix=f"{prefix}.to_lkk")
        self.to_lkv = BatchedReplicatedLinear(self.head_size, self.head_size, num_heads, bias=True, prefix=f"{prefix}.to_lkv")
        self.to_rqq = BatchedReplicatedLinear(self.head_size, self.head_size, num_heads, bias=True, prefix=f"{prefix}.to_rqq")
        self.to_rqk = BatchedReplicatedLinear(self.head_size, self.head_size, num_heads, bias=True, prefix=f"{prefix}.to_rqk")
        self.to_rqv = BatchedReplicatedLinear(self.head_size, self.head_size, num_heads, bias=True, prefix=f"{prefix}.to_rqv")

    def rot_emb_flat_unflat(self, x, cos, sin, is_neox_style):
        return _apply_rotary_emb(x, cos, sin, is_neox_style=is_neox_style)

    def get_block_sizes(self, seq_len):
        if not self.use_dynamic:
            seq_len_per_frame = 1456
            return (seq_len_per_frame // 52, 52)
        else:
            factors = [(i, seq_len // i) for i in range(1, math.floor(math.sqrt(seq_len)) + 1) if seq_len % i == 0]
            # choose the pair closest to square where one factor is divisible by 52
            remaining = [f for f in factors if f[0] % 52 == 0 or f[1] % 52 == 0]
            assert len(remaining) > 0, "Cannot find block sizes divisible by 52"
            factors = remaining[-1]
            if factors[1] % 52 == 0:
                return factors
            else:
                return (factors[1], factors[0])

    def local_q(self, q, k, v, cos_j, sin_j):
        batch_size, num_frames, q_seq_len, _, _ = q.shape
        new_batch_size = batch_size * num_frames * q_seq_len

        q = q.view(new_batch_size, 1, self.num_heads, self.head_size)
        k = rearrange(k, 'b f i j h d -> (b f j) i h d')
        v = rearrange(v, 'b f i j h d -> (b f j) i h d')

        assert new_batch_size == k.size(0) and new_batch_size == v.size(0)

        x = flash_attention(q, k, v, softmax_scale=self.softmax_scale)
        x = rearrange(x, '(b f j) 1 h d -> b (f j) h d', b=batch_size, f=num_frames)
        x = self.rot_emb_flat_unflat(x, cos_j, sin_j, is_neox_style=False)

        return rearrange(x, 'b (f j) h d -> b f j h d', f=num_frames)
    
    def local_k(self, q, k, v, cos_k, sin_k):
        batch_size, num_frames, q_seq_len, _, _ = q.shape
        new_batch_size = batch_size * num_frames * q_seq_len

        q = q.view(new_batch_size, 1, self.num_heads, self.head_size)
        k = k.view(new_batch_size, -1, self.num_heads, self.head_size)
        v = v.view(new_batch_size, -1, self.num_heads, self.head_size)

        x = flash_attention(q, k, v, softmax_scale=self.softmax_scale)
        x = rearrange(x, '(b f k) 1 h d -> b (f k) h d', b=batch_size, f=num_frames)
        x = self.rot_emb_flat_unflat(x, cos_k, sin_k, is_neox_style=False)

        return rearrange(x, 'b (f k) h d -> b f k h d', f=num_frames)

    @torch.compiler.disable
    def forward(
        self,
        lq: torch.Tensor,
        lk: torch.Tensor,
        rq: torch.Tensor,
        rk: torch.Tensor,
        v: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass for distributed attention.
        
        Args:
            q (torch.Tensor): Query tensor [batch_size, seq_len, num_heads, head_dim]
            k (torch.Tensor): Key tensor [batch_size, seq_len, num_heads, head_dim]
            v (torch.Tensor): Value tensor [batch_size, seq_len, num_heads, head_dim]
            replicated_q (Optional[torch.Tensor]): Replicated query tensor, typically for text tokens
            replicated_k (Optional[torch.Tensor]): Replicated key tensor
            replicated_v (Optional[torch.Tensor]): Replicated value tensor
            
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: A tuple containing:
                - o (torch.Tensor): Output tensor after attention for the main sequence
                - replicated_o (Optional[torch.Tensor]): Output tensor for replicated tokens, if provided
        """
        # Check text tokens are not supported for VSA now
        assert lq.dim() == 4 and lk.dim() == 4 and rq.dim() == 4 and rk.dim() == 4 and v.dim() == 4, "Expected 4D tensor"
        assert get_sp_world_size() == 1, "Monarch attention does not support sequence parallelism for now"

        batch_size = lq.size(0)
        block_b1, block_b2 = self.get_block_sizes(lq.size(-3))
        
        cos, sin = freqs_cis
        lq = self.rot_emb_flat_unflat(lq, cos, sin, is_neox_style=False)

        lkq = lk.view(batch_size, -1, block_b1, block_b2, self.num_heads, self.head_size)[:, :, :, 0]
        lkq, _ = self.to_lkq(lkq)
        cos_k, sin_k = cos.unflatten(0, (-1, block_b1, block_b2))[:, :, 0].flatten(0, 1), sin.unflatten(0, (-1, block_b1, block_b2))[:, :, 0].flatten(0, 1)
        lkq = self.rot_emb_flat_unflat(lkq.view(batch_size, -1, self.num_heads, self.head_size), cos_k, sin_k, is_neox_style=False).view(batch_size, -1, block_b1, self.num_heads, self.head_size)
        lkk = self.rot_emb_flat_unflat(self.to_lkk(lk)[0], cos, sin, is_neox_style=False)
        lkv, _ = self.to_lkv(lk)

        lq = lq.view(batch_size, -1, block_b1, block_b2, self.num_heads, self.head_size)
        lkq = lkq.view(batch_size, -1, block_b1, self.num_heads, self.head_size)
        lkk = lkk.view(batch_size, -1, block_b1, block_b2, self.num_heads, self.head_size)
        lkv = lkv.view(batch_size, -1, block_b1, block_b2, self.num_heads, self.head_size)

        lk = self.local_k(lkq, lkk, lkv, cos_k, sin_k)

        L = torch.einsum('baijhd,bfkhd->bhafjik', lq, lk)

        rk = self.rot_emb_flat_unflat(rk, cos, sin, is_neox_style=False)

        rqq = rq.view(batch_size, -1, block_b1, block_b2, self.num_heads, self.head_size)[:, :, 0]
        rqq, _ = self.to_rqq(rqq)
        cos_j, sin_j = cos.unflatten(0, (-1, block_b1, block_b2))[:, 0].flatten(0, 1), sin.unflatten(0, (-1, block_b1, block_b2))[:, 0].flatten(0, 1)
        rqq = self.rot_emb_flat_unflat(rqq.view(batch_size, -1, self.num_heads, self.head_size), cos_j, sin_j, is_neox_style=False).view(batch_size, -1, block_b2, self.num_heads, self.head_size)
        rqk = self.rot_emb_flat_unflat(self.to_rqk(rq)[0], cos, sin, is_neox_style=False)
        rqv, _ = self.to_rqv(rq)

        rk = rk.view(batch_size, -1, block_b1, block_b2, self.num_heads, self.head_size)
        rqq = rqq.view(batch_size, -1, block_b2, self.num_heads, self.head_size)
        rqk = rqk.view(batch_size, -1, block_b1, block_b2, self.num_heads, self.head_size)
        rqv = rqv.view(batch_size, -1, block_b1, block_b2, self.num_heads, self.head_size)

        rq = self.local_q(rqq, rqk, rqv, cos_j, sin_j)

        R = torch.einsum('bajhd,bfklhd->bhafkjl', rq, rk)

        out = torch.einsum('bhafjik,bhafkjl->bhafijkl', L, R)
        out = rearrange(out, 'b h a f i j k l -> b h (a i j) (f k l)')
        out = torch.softmax(out * self.softmax_scale, dim=-1)
        return torch.einsum('bhsl,blhd->bshd', out, v)
