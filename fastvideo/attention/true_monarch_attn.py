import torch
import torch.nn.functional as F

class MonarchAttnImplicitFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, sm_scale, num_iters, eps):
        b, i, j, _, _ = Q.shape
        block_b1, block_b2 = i, j

        sm_scale_sqrt = sm_scale ** 0.5
        Q = Q * sm_scale_sqrt
        K = K * sm_scale_sqrt

        L = torch.eye(block_b1, device=Q.device, dtype=Q.dtype).view(1, 1, 1, block_b1, block_b1).expand(b, Q.size(-2), block_b2, block_b1, block_b1) # (b, h, j, k, i)

        with torch.no_grad():
            for _ in range(num_iters):
                aR = torch.einsum("bhjki,bijhd->bkjhd", L, Q)
                bR = torch.einsum("bkjhd,bklhd->bhkjl", aR, K)
                cR = torch.einsum("bhjki->bhkj", L).unsqueeze(-1)
                R = torch.softmax(bR / (cR + eps), dim=-1)

                aL = torch.einsum("bhkjl,bklhd->bjkhd", R, K)
                bL = torch.einsum("bjkhd,bijhd->bhjki", aL, Q)
                cL = torch.einsum("bhkjl->bhjk", torch.xlogy(R, R)).unsqueeze(-1)
                L = torch.softmax(bL - cL, dim=-2)

        ctx.save_for_backward(L, R, Q, K, V)
        ctx.eps = eps
        ctx.sm_scale_sqrt = sm_scale_sqrt

        out = torch.einsum("bhkjl,bklhd->bjkhd", R, V)
        out = torch.einsum("bhjki,bjkhd->bijhd", L, out)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        L_star, R_star, Q, K, V = ctx.saved_tensors
        eps = ctx.eps

        tmp = torch.einsum("bhkjl,bklhd->bjkhd", R_star, V)
        grad_L = torch.einsum("bijhd,bjkhd->bhjki", grad_out, tmp)

        grad_tmp = torch.einsum("bijhd,bhjki->bjkhd", grad_out, L_star)
        grad_V = torch.einsum("bhkjl,bjkhd->bklhd", R_star, grad_tmp)

        def T_state(L_in):
            aR = torch.einsum("bhjki,bijhd->bkjhd", L_in, Q)
            bR = torch.einsum("bkjhd,bklhd->bhkjl", aR, K)
            cR = torch.einsum("bhjki->bhkj", L_in).unsqueeze(-1)
            R_out = torch.softmax(bR / (cR + eps), dim=-1)
            aL = torch.einsum("bhkjl,bklhd->bjkhd", R_out, K)
            bL = torch.einsum("bjkhd,bijhd->bhjki", aL, Q)
            cL = torch.einsum("bhkjl->bhjk", torch.xlogy(R_out, R_out)).unsqueeze(-1)
            L_out = torch.softmax(bL - cL, dim=-2)
            return L_out

        uL = torch.zeros_like(grad_L)
        damping = 1.0
        max_solve_iters = 20
        for _ in range(max_solve_iters):
            Au = uL - torch.autograd.functional.vjp(T_state, L_star, v=uL, create_graph=False)[1]
            resid = grad_L - Au
            uL = uL + damping * resid
        
        def T_wrt_params(q, k):
            aR = torch.einsum("bhjki,bijhd->bkjhd", L_star, q)
            bR = torch.einsum("bkjhd,bklhd->bhkjl", aR, k)
            cR = torch.einsum("bhjki->bhkj", L_star).unsqueeze(-1)
            R_out = torch.softmax(bR / (cR + eps), dim=-1)

            aL = torch.einsum("bhkjl,bklhd->bjkhd", R_out, k)
            bL = torch.einsum("bjkhd,bijhd->bhjki", aL, q)
            cL = torch.einsum("bhkjl->bhjk", torch.xlogy(R_out, R_out)).unsqueeze(-1)
            L_out = torch.softmax(bL - cL, dim=-2)
            return L_out

        _, (grad_Q, grad_K) = torch.autograd.functional.vjp(
            T_wrt_params, (Q, K), v=uL, create_graph=False, strict=True
        )

        grad_Q = grad_Q * ctx.sm_scale_sqrt
        grad_K = grad_K * ctx.sm_scale_sqrt

        return grad_Q, grad_K, grad_V, None, None, None

monarch_attn = MonarchAttnImplicitFn.apply
__all__ = [
    'monarch_attn',
]