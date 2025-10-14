from einops import rearrange
import torch
import torch.nn.functional as F

class MonarchAttnImplicitFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, sm_scale, num_iters, eps):
        b, a, i, j, h, _ = Q.shape
        block_b1, block_b2 = i, j
        f = K.size(-5)

        sm_scale_sqrt = sm_scale ** 0.5
        Q = Q * sm_scale_sqrt
        K = K * sm_scale_sqrt

        L = torch.eye(block_b1, device=Q.device, dtype=Q.dtype).view(1, 1, 1, 1, 1, block_b1, block_b1).expand(b, h, a, f, block_b2, block_b1, block_b1) # (b, h, a, f, j, k, i)

        with torch.no_grad():
            for _ in range(num_iters):
                aR = torch.einsum("bhafjki,baijhd->bafkjhd", L, Q)
                bR = torch.einsum("bafkjhd,bfklhd->bhafkjl", aR, K)
                # cR = torch.einsum("bhjki->bhkj", L_star).unsqueeze(-1)
                # R = torch.softmax(bR / (cR + eps), dim=-1)
                cR = L.sum(dim=-1, dtype=torch.float32).unsqueeze(-1).clamp_min(eps).transpose(-2, -3) # (b, h, a, f, k, j, 1)
                z = bR.to(torch.float32) * (1.0 / (cR + eps)).clamp_max(1e4)
                z = z - z.amax(dim=-1, keepdim=True)
                R = torch.softmax(z, dim=-1).to(Q.dtype)

                aL = torch.einsum("bhafkjl,bfklhd->bafjkhd", R, K)
                bL = torch.einsum("bafjkhd,baijhd->bhafjki", aL, Q)
                # cL = torch.einsum("bhkjl->bhjk", torch.xlogy(R, R)).unsqueeze(-1)
                logz = torch.logsumexp(z, dim=-1, keepdim=True) # (b, h, a, f, k, j, 1)
                cL = (R * (z - logz)).sum(dim=-1, keepdim=True).transpose(-2, -3) # (b, h, a, f, j, k, 1)
                L = rearrange(bL - cL, "b h a f j k i -> b h a j i (f k)")
                L = torch.softmax(L, dim=-1).to(Q.dtype)
                L = rearrange(L, "b h a j i (f k) -> b h a f j k i", f=f, k=block_b1)

        ctx.save_for_backward(L, R, Q, K, V)
        ctx.eps = eps
        ctx.sm_scale_sqrt = sm_scale_sqrt

        out = torch.einsum("bhafkjl,bfklhd->bafjkhd", R, V)
        out = torch.einsum("bhafjki,bafjkhd->baijhd", L, out)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        L_star, R_star, Q, K, V = ctx.saved_tensors
        eps = ctx.eps

        b, a, i, j, h, _ = Q.shape
        block_b1, block_b2 = i, j
        f = K.size(-5)

        grad_tmp = torch.einsum("baijhd,bhafjki->bafjkhd", grad_out, L_star)
        grad_V = torch.einsum("bhafkjl,bafjkhd->bfklhd", R_star, grad_tmp)
        grad_R = torch.einsum("bafjkhd,bfklhd->bhafkjl", grad_tmp, V)

        def R_from_QK(Q_in, K_in):
            aR = torch.einsum("bhafjki,baijhd->bafkjhd", L_star, Q_in)
            bR = torch.einsum("bafkjhd,bfklhd->bhafkjl", aR, K_in)
            # cR = torch.einsum("bhjki->bhkj", L_star).unsqueeze(-1)
            # R = torch.softmax(bR / (cR + eps), dim=-1)
            cR = L_star.sum(dim=-1, dtype=torch.float32).unsqueeze(-1).clamp_min(eps).transpose(-2, -3) # (b, h, a, f, k, j, 1)
            z = bR.to(torch.float32) * (1.0 / (cR + eps)).clamp_max(1e4)
            z = z - z.amax(dim=-1, keepdim=True)
            R = torch.softmax(z, dim=-1)
            return R.to(Q_in.dtype)

        _, (grad_Q, grad_K) = torch.autograd.functional.vjp(R_from_QK, (Q, K), v=grad_R, create_graph=False, strict=True)

        def O_from_L(L_in):
            aR = torch.einsum("bhafjki,baijhd->bafkjhd", L_in, Q)
            bR = torch.einsum("bafkjhd,bfklhd->bhafkjl", aR, K)
            # cR = torch.einsum("bhjki->bhkj", L_in).unsqueeze(-1)
            # R = torch.softmax(bR / (cR + eps), dim=-1)
            cR = L_in.sum(dim=-1, dtype=torch.float32).unsqueeze(-1).clamp_min(eps).transpose(-2, -3) # (b, h, a, f, k, j, 1)
            z = bR.to(torch.float32) * (1.0 / (cR + eps)).clamp_max(1e4)
            z = z - z.amax(dim=-1, keepdim=True)
            R = torch.softmax(z, dim=-1).to(L_in.dtype)

            out = torch.einsum("bhafkjl,bfklhd->bafjkhd", R, V)
            out = torch.einsum("bhafjki,bafjkhd->baijhd", L_in, out)
            return out
        
        _, grad_L = torch.autograd.functional.vjp(O_from_L, L_star, v=grad_out, create_graph=False)

        def L_from_L(L_in):
            aR = torch.einsum("bhafjki,baijhd->bafkjhd", L_in, Q)
            bR = torch.einsum("bafkjhd,bfklhd->bhafkjl", aR, K)
            # cR = torch.einsum("bhjki->bhkj", L_in).unsqueeze(-1)
            # R_out = torch.softmax(bR / (cR + eps), dim=-1)
            cR = L_in.sum(dim=-1, dtype=torch.float32).unsqueeze(-1).clamp_min(eps).transpose(-2, -3) # (b, h, a, f, k, j, 1)
            z = bR.to(torch.float32) * (1.0 / (cR + eps)).clamp_max(1e4)
            z = z - z.amax(dim=-1, keepdim=True)
            R_out = torch.softmax(z, dim=-1).to(L_in.dtype)

            aL = torch.einsum("bhafkjl,bfklhd->bafjkhd", R_out, K)
            bL = torch.einsum("bafjkhd,baijhd->bhafjki", aL, Q)
            # cL = torch.einsum("bhkjl->bhjk", torch.xlogy(R_out, R_out)).unsqueeze(-1)
            logz = torch.logsumexp(z, dim=-1, keepdim=True) # (b, h, k, j, 1)
            cL = (R_out * (z - logz)).sum(dim=-1, keepdim=True).transpose(-2, -3) # (b, h, a, f, j, k, 1)
            # L_out = torch.softmax(bL - cL, dim=-2).to(L_in.dtype)
            L_out = rearrange(bL - cL, "b h a f j k i -> b h a j i (f k)")
            L_out = torch.softmax(L_out, dim=-1).to(Q.dtype)
            L_out = rearrange(L_out, "b h a j i (f k) -> b h a f j k i", f=f, k=block_b1)
            return L_out

        u = torch.zeros_like(grad_L)
        r = grad_L.clone()
        for _ in range(1):
            u = u + r
            _, r = torch.autograd.functional.vjp(L_from_L, L_star, v=r, create_graph=False)
        
        def L_from_QK(Q, K):
            aR = torch.einsum("bhafjki,baijhd->bafkjhd", L_star, Q)
            bR = torch.einsum("bafkjhd,bfklhd->bhafkjl", aR, K)
            # cR = torch.einsum("bhjki->bhkj", L_star).unsqueeze(-1)
            # R_out = torch.softmax(bR / (cR + eps), dim=-1)
            cR = L_star.sum(dim=-1, dtype=torch.float32).unsqueeze(-1).clamp_min(eps).transpose(-2, -3) # (b, h, a, f, k, j, 1)
            z = bR.to(torch.float32) * (1.0 / (cR + eps)).clamp_max(1e4)
            z = z - z.amax(dim=-1, keepdim=True)
            R_out = torch.softmax(z, dim=-1).to(Q.dtype)

            aL = torch.einsum("bhafkjl,bfklhd->bafjkhd", R_out, K)
            bL = torch.einsum("bafjkhd,baijhd->bhafjki", aL, Q)
            # cL = torch.einsum("bhkjl->bhjk", torch.xlogy(R_out, R_out)).unsqueeze(-1)
            logz = torch.logsumexp(z, dim=-1, keepdim=True) # (b, h, k, j, 1)
            cL = (R_out * (z - logz)).sum(dim=-1, keepdim=True).transpose(-2, -3) # (b, h, a, f, j, k, 1)
            # L_out = torch.softmax(bL - cL, dim=-2).to(Q.dtype)
            L_out = rearrange(bL - cL, "b h a f j k i -> b h a j i (f k)")
            L_out = torch.softmax(L_out, dim=-1).to(Q.dtype)
            L_out = rearrange(L_out, "b h a j i (f k) -> b h a f j k i", f=f, k=block_b1)
            return L_out

        _, (grad_Q_L, grad_K_L) = torch.autograd.functional.vjp(
            L_from_QK, (Q, K), v=u, create_graph=False, strict=True
        )

        grad_Q += grad_Q_L
        grad_K += grad_K_L

        grad_Q = grad_Q * ctx.sm_scale_sqrt
        grad_K = grad_K * ctx.sm_scale_sqrt

        return grad_Q, grad_K, grad_V, None, None, None

monarch_attn = MonarchAttnImplicitFn.apply
__all__ = [
    'monarch_attn',
]