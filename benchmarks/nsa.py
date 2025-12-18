#!/usr/bin/env python3
"""NSA (Native Sparse Attention) benchmark wrapper for multiple backends"""
import torch
import tilelang
import tilelang.language as T
from .utils import get_target_and_backend


def native_sparse_attention_kernel(batch, heads, seq_len, dim, is_causal, scale, block_size, groups, selected_blocks):
    """Define NSA kernel using TileLang - from deepseek_nsa example"""
    if scale is None:
        scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    else:
        scale = scale * 1.44269504  # log2(e)

    head_kv = heads // groups
    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len, head_kv, dim]
    block_indices_shape = [batch, seq_len, head_kv, selected_blocks]
    block_indices_dtype = T.int32
    dtype = T.float16
    accum_dtype = T.float32
    block_S = block_size
    block_T = min(128, tilelang.math.next_power_of_2(dim))

    NK = tilelang.cdiv(dim, block_T)
    NV = tilelang.cdiv(dim, block_T)
    assert NK == 1, "The key dimension can not be larger than 256"

    S = selected_blocks
    G = groups
    BS = block_S
    BK = BV = block_T
    num_stages = 2
    threads = 32

    @T.prim_func
    def nsa_kernel(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(kv_shape, dtype),
        V: T.Tensor(kv_shape, dtype),
        BlockIndices: T.Tensor(block_indices_shape, block_indices_dtype),
        Output: T.Tensor(q_shape, dtype),
    ):
        with T.Kernel(seq_len, NV, batch * head_kv, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([G, BK], dtype)
            K_shared = T.alloc_shared([BS, BK], dtype)
            V_shared = T.alloc_shared([BS, BV], dtype)
            O_shared = T.alloc_shared([G, BV], dtype)

            acc_s = T.alloc_fragment([G, BS], accum_dtype)
            acc_s_cast = T.alloc_fragment([G, BS], dtype)
            acc_o = T.alloc_fragment([G, BV], accum_dtype)
            scores_max = T.alloc_fragment([G], accum_dtype)
            scores_max_prev = T.alloc_fragment([G], accum_dtype)
            scores_scale = T.alloc_fragment([G], accum_dtype)
            scores_sum = T.alloc_fragment([G], accum_dtype)
            logsum = T.alloc_fragment([G], accum_dtype)

            i_t, i_v, i_bh = bx, by, bz
            i_b, i_h = i_bh // head_kv, i_bh % head_kv

            NS = S
            T.copy(Q[i_b, i_t, i_h * G : (i_h + 1) * G, :], Q_shared)

            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            for i in T.Pipelined(NS, num_stages=num_stages):
                i_s = BlockIndices[i_b, i_t, i_h, i] * BS
                if i_s <= i_t and i_s >= 0:
                    T.copy(K[i_b, i_s : i_s + BS, i_h, :], K_shared)

                    if is_causal:
                        for i, j in T.Parallel(G, BS):
                            acc_s[i, j] = T.if_then_else(i_t >= (i_s + j), 0, -T.infinity(acc_s.dtype))
                    else:
                        T.clear(acc_s)

                    T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                    # Softmax
                    T.copy(scores_max, scores_max_prev)
                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.reduce_max(acc_s, scores_max, dim=1, clear=True)
                    for i in T.Parallel(G):
                        scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                    for i, j in T.Parallel(G, BS):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for i in T.Parallel(G):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                    T.copy(acc_s, acc_s_cast)

                    # Rescale
                    for i, j in T.Parallel(G, BV):
                        acc_o[i, j] *= scores_scale[i]

                    # V * softmax(Q * K)
                    T.copy(V[i_b, i_s : i_s + BS, i_h, i_v * BV : (i_v + 1) * BV], V_shared)
                    T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            for i, j in T.Parallel(G, BV):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output[i_b, i_t, i_h * G : (i_h + 1) * G, i_v * BV : (i_v + 1) * BV])

    return nsa_kernel


def pytorch_nsa(Q, K, V, block_indices, block_size, scale, is_causal=True):
    """PyTorch reference implementation of NSA"""
    B, SEQ_LEN, HQ, D = Q.shape
    _, _, H, _ = K.shape
    G = HQ // H
    
    if scale is None:
        scale = (1.0 / D) ** 0.5
    
    # Reshape for group query attention
    Q_reshaped = Q.view(B, SEQ_LEN, H, G, D)
    
    outputs = []
    for b in range(B):
        batch_out = []
        for t in range(SEQ_LEN):
            token_out = []
            for h in range(H):
                # Get block indices for this position
                indices = block_indices[b, t, h]
                valid_mask = indices < SEQ_LEN
                valid_indices = indices[valid_mask]
                
                if len(valid_indices) == 0:
                    token_out.append(torch.zeros(G, D, device=Q.device, dtype=Q.dtype))
                    continue
                
                # Gather K, V blocks
                block_starts = valid_indices * block_size
                k_blocks = []
                v_blocks = []
                for start in block_starts:
                    end = min(start + block_size, SEQ_LEN)
                    k_blocks.append(K[b, start:end, h, :])
                    v_blocks.append(V[b, start:end, h, :])
                
                # Pad to block_size if needed
                k_concat = torch.cat([
                    torch.nn.functional.pad(kb, (0, 0, 0, block_size - kb.shape[0]))
                    for kb in k_blocks
                ], dim=0)  # [num_blocks * block_size, D]
                v_concat = torch.cat([
                    torch.nn.functional.pad(vb, (0, 0, 0, block_size - vb.shape[0]))
                    for vb in v_blocks
                ], dim=0)
                
                # Compute attention for all groups
                q_groups = Q_reshaped[b, t, h, :, :]  # [G, D]
                scores = torch.matmul(q_groups, k_concat.t()) * scale  # [G, num_blocks * block_size]
                
                # Apply causal mask
                if is_causal:
                    for g in range(G):
                        for idx, start in enumerate(block_starts):
                            for j in range(block_size):
                                pos = start + j
                                if pos > t:
                                    scores[g, idx * block_size + j] = float('-inf')
                
                attn_weights = torch.softmax(scores, dim=-1)
                output = torch.matmul(attn_weights, v_concat)  # [G, D]
                token_out.append(output)
            
            batch_out.append(torch.stack(token_out, dim=0))  # [H, G, D]
        outputs.append(torch.stack(batch_out, dim=0))  # [SEQ_LEN, H, G, D]
    
    result = torch.stack(outputs, dim=0)  # [B, SEQ_LEN, H, G, D]
    return result.reshape(B, SEQ_LEN, HQ, D)


def get_benchmark_kernel(backend, arch, batch, heads, seq_len, dim, 
                        block_size=32, groups=1, selected_blocks=16, 
                        scale=None, is_causal=True, **kwargs):
    """
    Get compiled NSA kernel for specified backend
    
    Returns:
        (callable_kernel, theoretical_flops)
    """
    # Calculate actual FLOPs for sparse attention
    # Per token, per head_kv: selected_blocks iterations of:
    #   - QK^T: [G×dim] × [dim×BS] = 2×G×dim×BS
    #   - Attn×V: [G×BS] × [BS×dim] = 2×G×BS×dim
    # Total per token per head_kv: selected_blocks × 4 × G × dim × BS
    head_kv = heads // groups
    G = groups
    flops = batch * seq_len * head_kv * selected_blocks * 4 * G * dim * block_size
    
    if backend == "pytorch":
        def torch_nsa(q, k, v, block_indices, Output):
            return pytorch_nsa(q, k, v, block_indices, block_size, scale, is_causal)
        return torch_nsa, flops
    
    # CuTeDSL or CUDA backend
    kernel = native_sparse_attention_kernel(
        batch, heads, seq_len, dim, is_causal, scale,
        block_size, groups, selected_blocks
    )
    
    target = "cutedsl" if backend == "cutedsl" else "cuda"
    target_str, exec_backend = get_target_and_backend(target, arch)
    
    compile_kwargs = dict(
        target=target_str, 
        execution_backend=exec_backend,
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        }
    )
    
    compiled = tilelang.compile(kernel, **compile_kwargs)
    return compiled, flops


def prepare_inputs(batch, heads, seq_len, dim, block_size=32, 
                   groups=1, selected_blocks=16, **kwargs):
    """Prepare input and output tensors for NSA"""
    head_kv = heads // groups
    
    Q = torch.randn(batch, seq_len, heads, dim, device="cuda", dtype=torch.float16)
    K = torch.randn(batch, seq_len, head_kv, dim, device="cuda", dtype=torch.float16)
    V = torch.randn(batch, seq_len, head_kv, dim, device="cuda", dtype=torch.float16)
    Output = torch.empty(batch, seq_len, heads, dim, device="cuda", dtype=torch.float16)
    
    # Generate random block indices
    block_indices = torch.full((batch, seq_len, head_kv, selected_blocks), 
                               seq_len, dtype=torch.long, device="cuda")
    for b in range(batch):
        for t in range(seq_len):
            for h in range(head_kv):
                max_blocks = max(1, t // block_size)
                num_select = min(selected_blocks, max_blocks)
                indices = torch.randperm(max_blocks, device="cuda")[:num_select]
                block_indices[b, t, h, :num_select] = indices
    
    block_indices = block_indices.sort(-1)[0]
    
    return (Q, K, V, block_indices.to(torch.int32), Output)


def verify_correctness(kernel, inputs, block_size=32, scale=None, is_causal=True, **kwargs):
    """Verify kernel correctness against PyTorch reference"""
    Q, K, V, block_indices, Output = inputs
    
    try:
        # Run kernel (modifies Output in-place for tilelang, returns for pytorch)
        result = kernel(Q, K, V, block_indices, Output)
        if result is None:
            result = Output
        
        # Run PyTorch reference
        Output_ref = pytorch_nsa(Q, K, V, block_indices, block_size, scale, is_causal)
        
        # Check shapes
        if result.shape != Output_ref.shape:
            return False, float('inf')
        
        # Check for NaNs/Infs
        if torch.isnan(result).any() or torch.isinf(result).any():
            return False, float('inf')
        
        # Calculate max difference
        max_diff = torch.max(torch.abs(result.cpu() - Output_ref.cpu())).item()
        
        # Relaxed tolerance for sparse attention
        try:
            torch.testing.assert_close(result.cpu(), Output_ref.cpu(), atol=1e-2, rtol=1e-2)
            return True, max_diff
        except AssertionError:
            return False, max_diff
            
    except Exception as e:
        print(f"    Verification error: {e}")
        return False, float('inf')
