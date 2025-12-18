#!/usr/bin/env python3
"""
CuTeDSL Benchmark Runner
Systematically benchmark GEMM and MHA kernels across different backends
"""
import json
import time
import argparse
from pathlib import Path
import torch
import cuda.bindings.driver as cuda
from benchmarks.utils import get_gpu_info


def benchmark_single(kernel_func, args, warmup=10, iters=100):
    """
    Time a single kernel using CUDA events
    
    Returns:
        latency_ms: average latency (milliseconds)
    """
    torch_stream = torch.cuda.current_stream()
    current_stream = cuda.CUstream(torch_stream.cuda_stream)
    
    start_event = cuda.cuEventCreate(cuda.CUevent_flags.CU_EVENT_DEFAULT)[1]
    end_event = cuda.cuEventCreate(cuda.CUevent_flags.CU_EVENT_DEFAULT)[1]
    
    # Warmup
    for _ in range(warmup):
        kernel_func(*args)
    
    # Benchmark
    cuda.cuEventRecord(start_event, current_stream)
    for _ in range(iters):
        kernel_func(*args)
    cuda.cuEventRecord(end_event, current_stream)
    cuda.cuEventSynchronize(end_event)
    
    elapsed_ms = cuda.cuEventElapsedTime(start_event, end_event)[1]
    
    cuda.cuEventDestroy(start_event)
    cuda.cuEventDestroy(end_event)
    
    return elapsed_ms / iters


def run_benchmark(kernel_name, backend, gpu_info, problem_size, warmup=10, iters=100, verify=False):
    """
    Run a single benchmark configuration
    
    Returns:
        result_dict or None if failed
    """
    # Dynamically import the corresponding kernel module
    if kernel_name == "gemm":
        from benchmarks import gemm as mod
    elif kernel_name == "mha":
        from benchmarks import mha as mod
    elif kernel_name == "nsa":
        from benchmarks import nsa as mod
    else:
        raise ValueError(f"Unknown kernel: {kernel_name}")
    
    print(f"  [{kernel_name}] {backend} {problem_size} ... ", end="", flush=True)
    
    try:
        # Compile kernel
        compile_start = time.time()
        kernel, flops = mod.get_benchmark_kernel(backend, gpu_info["arch"], **problem_size)
        compile_time = time.time() - compile_start
        
        # Prepare inputs
        inputs = mod.prepare_inputs(**problem_size)
        
        # Verify correctness (optional)
        if verify:
            correct, max_diff = mod.verify_correctness(kernel, inputs, **problem_size)
            if not correct:
                print(f"FAILED (correctness error, max_diff={max_diff:.6f})")
                return None
        
        # Benchmark
        latency = benchmark_single(kernel, inputs, warmup=warmup, iters=iters)
        
        # Calculate TFLOPS only if flops is provided
        if flops is not None:
            tflops = (flops / (latency / 1000)) / 1e12
        else:
            tflops = None
        
        print(f"OK ({latency:.4f} ms" + (f", {tflops:.2f} TFLOPS" if tflops else "") + ")")
        
        result = {
            "kernel": kernel_name,
            "backend": backend,
            "arch": gpu_info["arch"],
            "gpu_model": gpu_info["model"],
            "compute_capability": gpu_info["compute_capability"],
            "problem_size": problem_size,
            "latency_ms": latency,
            "compilation_time": compile_time
        }
        
        # Add TFLOPS only if available
        if tflops is not None:
            result["tflops"] = tflops
        
        return result
        
    except Exception as e:
        print(f"FAILED ({type(e).__name__}: {str(e)[:50]})")
        return None


def load_existing_results(output_path):
    """Load existing result file"""
    if output_path.exists():
        try:
            with open(output_path) as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading existing results: {e}")
            return []
    return []


def main():
    parser = argparse.ArgumentParser(description="CuTeDSL Benchmark Runner")
    parser.add_argument("--config", default="configs/default.json", help="Config file path")
    parser.add_argument("--output", default=None, help="Output file path (default: auto-generate)")
    parser.add_argument("--append", action="store_true", help="Append to existing output file")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=10, help="Benchmark iterations")
    parser.add_argument("--verify", action="store_true", help="Verify correctness")
    parser.add_argument("--kernels", nargs="+", default=["gemm", "mha", "nsa"], 
                       help="Kernels to benchmark")
    parser.add_argument("--backends", nargs="+", default=["cutedsl", "cuda", "pytorch"],
                       help="Backends to benchmark")
    args = parser.parse_args()
    
    # Get complete GPU information
    gpu_info = get_gpu_info()
    print("GPU Information:")
    print(f"  Model: {gpu_info['model']}")
    print(f"  Architecture: {gpu_info['arch']} (Compute {gpu_info['compute_capability']})")
    print(f"Configuration: warmup={args.warmup}, iters={args.iters}")
    print("")
    
    # Determine output file path
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        # Use simplified GPU model name (remove spaces and special characters)
        gpu_short = gpu_info["model"].replace(" ", "_").replace("/", "_")[:20]
        output_path = Path(f"results/{gpu_info['arch']}_{gpu_short}_{timestamp}.json")
    
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    # Load existing results if in append mode
    if args.append:
        existing_results = load_existing_results(output_path)
        print(f"Appending to existing file with {len(existing_results)} results")
        print("")
    else:
        existing_results = []
    
    # Load configuration
    config = json.load(open(args.config))
    
    # Run all benchmarks
    results = existing_results.copy()
    total = sum(len(config.get(k, [])) for k in args.kernels) * len(args.backends)
    current = 0
    new_results = 0
    
    for kernel_name in args.kernels:
        if kernel_name not in config:
            print(f"Warning: No config for kernel '{kernel_name}', skipping")
            continue
            
        print(f"=== Benchmarking {kernel_name.upper()} ===")
        
        for problem_size in config[kernel_name]:
            for backend in args.backends:
                current += 1
                print(f"[{current}/{total}] ", end="")
                
                result = run_benchmark(
                    kernel_name, backend, gpu_info, problem_size,
                    warmup=args.warmup, iters=args.iters, verify=args.verify
                )
                
                if result is not None:
                    results.append(result)
                    new_results += 1
        
        print("")
    
    # Save results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print("✓ Benchmark complete!")
    print(f"  New results: {new_results}/{total}")
    print(f"  Total in file: {len(results)}")
    print(f"  Results saved to: {output_path}")
    print("\nTo visualize:")
    print(f"  python visualize.py {output_path}")


if __name__ == "__main__":
    main()
