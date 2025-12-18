#!/usr/bin/env python3
"""
Visualize compilation benchmark results
"""
import json
import sys
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def plot_compilation_results(json_path, output_dir=None):
    """Plot compilation time comparison"""
    # Load data
    with open(json_path) as f:
        data = json.load(f)
    
    if not data:
        print("Error: No data in result file")
        return
    
    df = pd.DataFrame(data)
    
    # Extract system info
    gpu_model = df['gpu_model'].iloc[0]
    cpu_model = df['cpu_model'].iloc[0]
    arch = df['arch'].iloc[0]
    
    # Determine output directory
    if output_dir is None:
        output_dir = Path(json_path).parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create safe filename suffix
    gpu_safe = gpu_model.replace(" ", "_").replace("/", "_")[:20]
    cpu_safe = cpu_model.split('@')[0].strip().replace(" ", "_")[:30]
    
    print("Visualizing compilation benchmark:")
    print(f"  GPU: {gpu_model} ({arch})")
    print(f"  CPU: {cpu_model}")
    print(f"  Kernels: {df['kernel'].unique().tolist()}")
    print(f"  Backends: {df['backend'].unique().tolist()}")
    print("")
    
    # Plot for each kernel
    for kernel in df['kernel'].unique():
        kernel_df = df[df['kernel'] == kernel]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"{kernel.upper()} Compilation Time\n{gpu_model} + {cpu_model.split('@')[0]}", 
                    fontsize=13, fontweight='bold')
        
        # Average compilation time
        ax1.bar(kernel_df['backend'], kernel_df['compile_time_avg'], 
               yerr=[kernel_df['compile_time_avg'] - kernel_df['compile_time_min'],
                     kernel_df['compile_time_max'] - kernel_df['compile_time_avg']],
               capsize=5, alpha=0.7, color=['#2E86AB', '#A23B72'])
        ax1.set_ylabel("Compilation Time (seconds)", fontsize=11)
        ax1.set_xlabel("Backend", fontsize=11)
        ax1.set_title("Average Time to First Execution", fontsize=12)
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (backend, row) in enumerate(kernel_df.iterrows()):
            ax1.text(i, row['compile_time_avg'], f"{row['compile_time_avg']:.2f}s",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Speedup comparison (relative to slower backend)
        if len(kernel_df) >= 2:
            times = kernel_df.set_index('backend')['compile_time_avg']
            baseline = times.max()
            speedups = baseline / times
            
            ax2.bar(speedups.index, speedups.values, alpha=0.7, 
                   color=['#06A77D' if s > 1 else '#D95F69' for s in speedups.values])
            ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.7)
            ax2.set_ylabel("Speedup", fontsize=11)
            ax2.set_xlabel("Backend", fontsize=11)
            ax2.set_title(f"Compilation Speedup (vs {times.idxmax()})", fontsize=12)
            ax2.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for i, (backend, speedup) in enumerate(speedups.items()):
                ax2.text(i, speedup, f"{speedup:.2f}x",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'Need at least 2 backends\nfor speedup comparison',
                    ha='center', va='center', fontsize=11, transform=ax2.transAxes)
            ax2.axis('off')
        
        plt.tight_layout()
        
        # Save plot
        output_path = output_dir / f"compile_{kernel}_{arch}_{gpu_safe}_{cpu_safe}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_path}")
        plt.close()
    
    # Summary plot: all kernels together
    if len(df['kernel'].unique()) > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle(f"Compilation Time Summary\n{gpu_model} + {cpu_model.split('@')[0]}", 
                    fontsize=13, fontweight='bold')
        
        pivot = df.pivot_table(
            index='kernel',
            columns='backend',
            values='compile_time_avg',
            aggfunc='first'
        )
        
        pivot.plot(kind='bar', ax=ax, alpha=0.7, width=0.7)
        ax.set_ylabel("Compilation Time (seconds)", fontsize=11)
        ax.set_xlabel("Kernel", fontsize=11)
        ax.set_title("Time to First Execution", fontsize=12)
        ax.legend(title="Backend", fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=0)
        plt.tight_layout()
        
        output_path = output_dir / f"compile_summary_{arch}_{gpu_safe}_{cpu_safe}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_path}")
        plt.close()
    
    print("")
    print("✓ Visualization complete!")


def main():
    parser = argparse.ArgumentParser(description="Visualize compilation benchmark results")
    parser.add_argument("result_file", help="Path to JSON result file")
    parser.add_argument("--output", default=None, help="Output directory (default: same as input)")
    args = parser.parse_args()
    
    result_path = Path(args.result_file)
    if not result_path.exists():
        print(f"Error: Result file not found: {result_path}")
        sys.exit(1)
    
    plot_compilation_results(result_path, args.output)


if __name__ == "__main__":
    main()
