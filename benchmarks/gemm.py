#!/usr/bin/env python3
"""GEMM benchmark wrapper for multiple backends"""
import torch
import tilelang
import tilelang.language as T
from .utils import get_target_and_backend


def matmul(M, N, K, block_M, block_N, block_K, dtype, accum_dtype, threads):
    """Define GEMM kernel using TileLang"""
    @T.prim_func
    def gemm(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm


def get_benchmark_kernel(backend, arch, M, N, K, dtype="float16", **kwargs):
    """
    Get compiled GEMM kernel for specified backend
    
    Returns:
        (callable_kernel, theoretical_flops)
    """
    flops = 2 * M * N * K
    
    if backend == "pytorch":
        def torch_gemm(a, b, c):
            return torch.matmul(a, b)
        return torch_gemm, flops
    
    # CuTeDSL or CUDA backend
    kernel = matmul(M, N, K, block_M=128, block_N=128, block_K=32, 
                    dtype=dtype, accum_dtype="float32", threads=128)
    
    target = "cutedsl" if backend == "cutedsl" else "cuda"
    target_str, exec_backend = get_target_and_backend(target, arch)
    compiled = tilelang.compile(kernel, target=target_str, 
                                execution_backend=exec_backend, pass_configs={
                                    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
                                })
    return compiled, flops


def prepare_inputs(M, N, K, dtype="float16", **kwargs):
    """Prepare input and output tensors for GEMM"""
    a = torch.randn(M, K, device="cuda", dtype=getattr(torch, dtype))
    b = torch.randn(K, N, device="cuda", dtype=getattr(torch, dtype))
    c = torch.empty(M, N, device="cuda", dtype=getattr(torch, dtype))
    return (a, b, c)


def verify_correctness(kernel, inputs, **kwargs):
    """Verify kernel correctness against PyTorch"""
    a, b, c = inputs
    dtype = kwargs.get('dtype', 'float16')
    
    # Run kernel (modifies c in-place for tilelang, returns for pytorch)
    result = kernel(a, b, c)
    if result is None:
        result = c
    
    c_ref = torch.matmul(a, b).to(getattr(torch, dtype))
    max_diff = torch.max(torch.abs(result.cpu() - c_ref.cpu())).item()
    try:
        torch.testing.assert_close(result.cpu(), c_ref.cpu(), atol=1e-3, rtol=1e-3)
        return True, max_diff
    except AssertionError:
        return False, max_diff
