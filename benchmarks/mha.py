#!/usr/bin/env python3
"""MHA (Multi-Head Attention) benchmark wrapper for multiple backends"""
import torch
import torch.nn.functional as F
import tilelang
import tilelang.language as T
from .utils import get_target_and_backend


def flashattn(batch, heads, seq_len, dim, is_causal, block_M, block_N, num_stages, threads):
    """Define FlashAttention kernel using TileLang"""
    scale = (1.0 / dim)**0.5 * 1.44269504  # log2(e)
    shape = [batch, seq_len, heads, dim]
    dtype = "float16"
    accum_dtype = "float"

    @T.macro
    def MMA0(K, Q_shared, K_shared, acc_s, k, bx, by, bz):
        T.copy(K[bz, k * block_N:(k + 1) * block_N, by, :], K_shared)
        if is_causal:
            for i, j in T.Parallel(block_M, block_N):
                acc_s[i, j] = T.if_then_else(bx * block_M + i >= k * block_N + j, 0,
                                             -T.infinity(acc_s.dtype))
        else:
            T.clear(acc_s)
        T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

    @T.macro
    def MMA1(V, V_shared, acc_s_cast, acc_o, k, by, bz):
        T.copy(V[bz, k * block_N:(k + 1) * block_N, by, :], V_shared)
        T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

    @T.macro
    def Softmax(acc_s, acc_s_cast, scores_max, scores_max_prev, scores_scale, scores_sum, logsum):
        T.copy(scores_max, scores_max_prev)
        T.fill(scores_max, -T.infinity(accum_dtype))
        T.reduce_max(acc_s, scores_max, dim=1, clear=False)
        for i in T.Parallel(block_M):
            scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
        for i, j in T.Parallel(block_M, block_N):
            acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
        T.reduce_sum(acc_s, scores_sum, dim=1)
        for i in T.Parallel(block_M):
            logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
        T.copy(acc_s, acc_s_cast)

    @T.macro
    def Rescale(acc_o, scores_scale):
        for i, j in T.Parallel(block_M, dim):
            acc_o[i, j] *= scores_scale[i]

    @T.prim_func
    def main(
            Q: T.Tensor(shape, dtype),
            K: T.Tensor(shape, dtype),
            V: T.Tensor(shape, dtype),
            Output: T.Tensor(shape, dtype),
    ):
        with T.Kernel(
                T.ceildiv(seq_len, block_M), heads, batch,
                threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)
            O_shared = T.alloc_shared([block_M, dim], dtype)
            acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
            acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_scale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)

            T.copy(Q[bz, bx * block_M:(bx + 1) * block_M, by, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = (
                T.min(T.ceildiv(seq_len, block_N), T.ceildiv(
                    (bx + 1) * block_M, block_N)) if is_causal else T.ceildiv(seq_len, block_N))

            for k in T.Pipelined(loop_range, num_stages=num_stages):
                MMA0(K, Q_shared, K_shared, acc_s, k, bx, by, bz)
                Softmax(acc_s, acc_s_cast, scores_max, scores_max_prev, scores_scale, scores_sum,
                        logsum)
                Rescale(acc_o, scores_scale)
                MMA1(V, V_shared, acc_s_cast, acc_o, k, by, bz)
            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output[bz, bx * block_M:(bx + 1) * block_M, by, :])

    return main


def ref_program(Q, K, V, is_causal):
    """PyTorch reference implementation"""
    dim = Q.size(-1)
    scores = torch.einsum('bqhd,bkhd->bhqk', Q, K)
    scores = scores / torch.sqrt(torch.tensor(dim, dtype=scores.dtype))
    if is_causal:
        seq_len = Q.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=scores.device))
        mask = mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.einsum('bhqk,bkhd->bqhd', attention_weights, V)
    return output


def get_benchmark_kernel(backend, arch, batch, heads, seq_len, dim, is_causal=False, **kwargs):
    """
    Get compiled MHA kernel for specified backend
    
    Returns:
        (callable_kernel, theoretical_flops)
    """
    # Calculate FLOPS
    flops_per_matmul = 2.0 * batch * heads * seq_len * seq_len * dim
    total_flops = 2 * flops_per_matmul
    if is_causal:
        total_flops *= 0.5
    
    if backend == "pytorch":
        def torch_mha(q, k, v, Output):
            return ref_program(q, k, v, is_causal)
        return torch_mha, total_flops
    
    # CuTeDSL or CUDA backend
    block_M = 64
    block_N = 64
    num_stages = 1
    threads = 128
    
    kernel = flashattn(batch, heads, seq_len, dim, is_causal, 
                      block_M, block_N, num_stages, threads)
    
    target = "cutedsl" if backend == "cutedsl" else "cuda"
    target_str, exec_backend = get_target_and_backend(target, arch)
    
    compile_kwargs = dict(
        target=target_str, 
        execution_backend=exec_backend,
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        }
    )
    
    compiled = tilelang.compile(kernel, **compile_kwargs)
    return compiled, total_flops


def prepare_inputs(batch, heads, seq_len, dim, **kwargs):
    """Prepare input and output tensors for MHA"""
    shape = [batch, seq_len, heads, dim]
    Q = torch.randn(shape, device="cuda", dtype=torch.float16)
    K = torch.randn(shape, device="cuda", dtype=torch.float16)
    V = torch.randn(shape, device="cuda", dtype=torch.float16)
    Output = torch.empty(shape, device="cuda", dtype=torch.float16)
    return (Q, K, V, Output)


def verify_correctness(kernel, inputs, **kwargs):
    """Verify kernel correctness against PyTorch"""
    Q, K, V, Output = inputs
    is_causal = kwargs.get('is_causal', False)
    
    # Run kernel (modifies Output in-place for tilelang, returns for pytorch)
    result = kernel(Q, K, V, Output)
    if result is None:
        result = Output
    
    Output_ref = ref_program(Q, K, V, is_causal)
    max_diff = torch.max(torch.abs(result.cpu() - Output_ref.cpu())).item()
    try:
        torch.testing.assert_close(result.cpu(), Output_ref.cpu(), atol=1e-2, rtol=1e-2)
        return True, max_diff
    except AssertionError:
        return False, max_diff
