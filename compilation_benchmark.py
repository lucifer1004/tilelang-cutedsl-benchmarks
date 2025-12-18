#!/usr/bin/env python3
"""
Compilation Benchmark - Measure time to first kernel execution
NOTE: This script should be called ONCE per measurement from bash to avoid in-memory caching
"""
import json
import time
import argparse
from pathlib import Path
import torch
from benchmarks.utils import get_gpu_info


def get_cpu_info():
    """Get CPU model name"""
    try:
        with open('/proc/cpuinfo', 'r') as f:
            for line in f:
                if 'model name' in line:
                    return line.split(':')[1].strip()
    except Exception as e:
        print(f"Error getting CPU name: {e}")
        return "Unknown CPU"


def benchmark_kernel_compilation(kernel_name, backend, arch, config):
    """
    Benchmark time to first execution (compilation + first run)
    This should be called in a fresh Python process to avoid in-memory caching
    
    Measures total time from cache clear to first successful execution,
    including any runtime compilation (e.g., CuTeDSL cubin compilation)
    """
    # Import here to avoid premature loading
    import tilelang
    tilelang.disable_cache()

    if kernel_name == "gemm":
        from benchmarks import gemm as mod
    elif kernel_name == "mha":
        from benchmarks import mha as mod
    elif kernel_name == "nsa":
        from benchmarks import nsa as mod
    else:
        raise ValueError(f"Unknown kernel: {kernel_name}")
    
    # Start timing: from compilation to first execution
    start_time = time.time()
    
    # Compile kernel
    kernel, _ = mod.get_benchmark_kernel(backend, arch, **config)
    
    # Prepare inputs (use minimal data for benchmark configs)
    inputs = mod.prepare_inputs(**config)
    
    # First execution (may trigger runtime compilation like cubin generation)
    _ = kernel(*inputs)
    torch.cuda.synchronize()
    
    # End timing
    total_time = time.time() - start_time
    
    return total_time


def main():
    parser = argparse.ArgumentParser(description="Compilation Benchmark - Single Measurement")
    parser.add_argument("--kernel", required=True, choices=["gemm", "mha", "nsa"],
                       help="Kernel to benchmark")
    parser.add_argument("--backend", required=True, choices=["cuda", "cutedsl"],
                       help="Backend to use")
    parser.add_argument("--config", required=True, help="JSON config for kernel parameters")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--run-id", default="0", help="Run ID for this measurement")
    args = parser.parse_args()
    
    # Parse config
    config = json.loads(args.config)
    
    # Get system info
    gpu_info = get_gpu_info()
    cpu_info = get_cpu_info()
    
    print(f"Compiling {args.kernel} with {args.backend} (run {args.run_id})...")
    print(f"  Config: {config}")
    
    try:
        compile_time = benchmark_kernel_compilation(
            args.kernel, args.backend, gpu_info['arch'], config
        )
        
        result = {
            "kernel": args.kernel,
            "backend": args.backend,
            "arch": gpu_info['arch'],
            "gpu_model": gpu_info['model'],
            "cpu_model": cpu_info,
            "compute_capability": gpu_info['compute_capability'],
            "config": config,
            "compile_time": compile_time,
            "run_id": args.run_id
        }
        
        # Save result
        output_path = Path(args.output)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        
        print(f"  ✓ Compilation time: {compile_time:.3f}s")
        print(f"  ✓ Result saved to: {output_path}")
        
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
