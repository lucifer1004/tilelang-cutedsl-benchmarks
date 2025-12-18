#!/usr/bin/env python3
"""
Visualize benchmark results using matplotlib
"""
import json
import sys
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def format_problem_size(size_dict):
    """Format problem size dict to a readable string"""
    if "M" in size_dict:
        # GEMM: M x N x K
        return f"{size_dict['M']}x{size_dict['N']}x{size_dict['K']}"
    elif "batch" in size_dict:
        # MHA: B x H x S x D
        return f"B{size_dict['batch']}_H{size_dict['heads']}_S{size_dict['seq_len']}_D{size_dict['dim']}"
    else:
        return str(size_dict)


def plot_kernel_results(kernel_df, output_dir):
    """Plot results for a single kernel"""
    kernel_name = kernel_df['kernel'].iloc[0]
    
    # Use GPU model instead of SM architecture
    if 'gpu_model' in kernel_df.columns and pd.notna(kernel_df['gpu_model'].iloc[0]):
        gpu_name = kernel_df['gpu_model'].iloc[0]
    else:
        gpu_name = kernel_df['arch'].iloc[0]  # fallback to arch
    
    # Add formatted problem_size column
    kernel_df['size_label'] = kernel_df['problem_size'].apply(format_problem_size)
    
    # Create charts
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{kernel_name.upper()} Performance on {gpu_name}", fontsize=14, fontweight='bold')
    
    # Latency comparison (lower is better)
    ax1 = axes[0]
    pivot_latency = kernel_df.pivot_table(
        index="size_label", 
        columns="backend", 
        values="latency_ms", 
        aggfunc="first"
    )
    pivot_latency.plot(kind="bar", ax=ax1)
    ax1.set_ylabel("Latency (ms)", fontsize=11)
    ax1.set_xlabel("Problem Size", fontsize=11)
    ax1.set_title("Latency Comparison (Lower is Better)", fontsize=12)
    ax1.legend(title="Backend", fontsize=9)
    ax1.grid(axis='y', alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Throughput comparison (higher is better) - only show when TFLOPS data is available
    ax2 = axes[1]
    if 'tflops' in kernel_df.columns and kernel_df['tflops'].notna().any():
        pivot_tflops = kernel_df.pivot_table(
            index="size_label",
            columns="backend",
            values="tflops",
            aggfunc="first"
        )
        pivot_tflops.plot(kind="bar", ax=ax2)
        ax2.set_ylabel("TFLOPS", fontsize=11)
        ax2.set_xlabel("Problem Size", fontsize=11)
        ax2.set_title("Throughput Comparison (Higher is Better)", fontsize=12)
        ax2.legend(title="Backend", fontsize=9)
        ax2.grid(axis='y', alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
    else:
        # If no TFLOPS data (like NSA), show explanation
        ax2.text(0.5, 0.5, 'TFLOPS not applicable\nfor this kernel',
                ha='center', va='center', fontsize=12, transform=ax2.transAxes)
        ax2.set_title("Throughput (N/A)", fontsize=12)
        ax2.axis('off')
    
    plt.tight_layout()
    
    # Save image - filename includes GPU model
    gpu_safe = gpu_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
    output_path = output_dir / f"{kernel_name}_comparison_{gpu_safe}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_speedup(df, output_dir):
    """Plot speedup relative to baseline (PyTorch for most, CUDA for NSA)"""
    # Only process kernels with TFLOPS data
    df_with_tflops = df[df['tflops'].notna()]
    if len(df_with_tflops) == 0:
        print("  Warning: No TFLOPS data available, skipping speedup plot")
        return
    
    # Use DataFrame with TFLOPS data
    df = df_with_tflops
    
    # Use GPU model instead of SM architecture
    if 'gpu_model' in df.columns and pd.notna(df['gpu_model'].iloc[0]):
        gpu_name = df['gpu_model'].iloc[0]
    else:
        gpu_name = df['arch'].iloc[0]  # fallback to arch
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate speedup for each kernel (NSA uses CUDA baseline, others use PyTorch)
    results = []
    baseline_info = []  # Track which baseline is used for each kernel
    
    for kernel in df['kernel'].unique():
        kernel_df = df[df['kernel'] == kernel].copy()
        kernel_df['size_label'] = kernel_df['problem_size'].apply(format_problem_size)
        
        # Choose baseline: CUDA for NSA, PyTorch for others
        if kernel == 'nsa':
            baseline_backend = 'cuda'
            compare_backends = ['cutedsl']
        else:
            baseline_backend = 'pytorch'
            compare_backends = ['cuda', 'cutedsl']
        
        baseline_df = kernel_df[kernel_df['backend'] == baseline_backend]
        if len(baseline_df) == 0:
            print(f"  Warning: No {baseline_backend} baseline for {kernel}, skipping")
            continue
        
        baseline_info.append(f"{kernel}: {baseline_backend}")
        
        # Calculate speedup of other backends relative to baseline
        for backend in compare_backends:
            if backend not in kernel_df['backend'].values:
                continue
            
            backend_df = kernel_df[kernel_df['backend'] == backend]
            
            # Merge on size_label
            merged = backend_df.merge(
                baseline_df, 
                on='size_label', 
                suffixes=(f'_{backend}', f'_{baseline_backend}')
            )
            
            for _, row in merged.iterrows():
                speedup = row[f'tflops_{backend}'] / row[f'tflops_{baseline_backend}']
                results.append({
                    'kernel': row[f'kernel_{backend}'],
                    'size': row['size_label'],
                    'backend': backend,
                    'speedup': speedup
                })
    
    if not results:
        print("  Warning: No speedup data available")
        return
    
    speedup_df = pd.DataFrame(results)
    speedup_df['label'] = speedup_df['kernel'] + "_" + speedup_df['size']
    
    # Create title with baseline info
    baseline_text = ", ".join(baseline_info)
    fig.suptitle(f"Speedup vs Baseline on {gpu_name}\n({baseline_text})", 
                 fontsize=14, fontweight='bold')
    
    # Plot grouped bar chart
    pivot_speedup = speedup_df.pivot_table(
        index='label',
        columns='backend',
        values='speedup',
        aggfunc='first'
    )
    
    pivot_speedup.plot(kind='bar', ax=ax)
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Baseline (1.0x)')
    ax.set_ylabel("Speedup vs Baseline", fontsize=11)
    ax.set_xlabel("Kernel_Size", fontsize=11)
    ax.set_title("Speedup Comparison (Higher is Better)", fontsize=12)
    ax.legend(title="Backend", fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Generate safe filename
    gpu_safe = gpu_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
    output_path = output_dir / f"speedup_comparison_{gpu_safe}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize benchmark results")
    parser.add_argument("result_file", help="Path to JSON result file")
    parser.add_argument("--output", default=None, help="Output directory (default: same as result file)")
    args = parser.parse_args()
    
    result_path = Path(args.result_file)
    if not result_path.exists():
        print(f"Error: Result file not found: {result_path}")
        sys.exit(1)
    
    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = result_path.parent
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load data
    print(f"Loading results from: {result_path}")
    with open(result_path) as f:
        data = json.load(f)
    
    if not data:
        print("Error: No data in result file")
        sys.exit(1)
    
    df = pd.DataFrame(data)
    print(f"  Loaded {len(df)} benchmark results")
    print(f"  Kernels: {df['kernel'].unique().tolist()}")
    print(f"  Backends: {df['backend'].unique().tolist()}")
    print("")
    
    # Generate charts for each kernel
    print("Generating plots...")
    for kernel in df['kernel'].unique():
        kernel_df = df[df['kernel'] == kernel]
        plot_kernel_results(kernel_df, output_dir)
    
    # Generate speedup chart
    plot_speedup(df, output_dir)
    
    print("")
    print("✓ Visualization complete!")


if __name__ == "__main__":
    main()
