#!/usr/bin/env python3
"""
Merge multiple benchmark result files into one
Usage: python merge_results.py result1.json result2.json -o merged.json
"""
import json
import argparse
from pathlib import Path


def merge_results(*files, output=None, deduplicate=True):
    """
    Merge multiple result files
    
    Args:
        files: Input JSON file paths
        output: Output file path
        deduplicate: Whether to deduplicate (based on kernel+backend+problem_size)
    """
    all_results = []
    
    for file in files:
        print(f"Loading {file} ... ", end="")
        with open(file) as f:
            data = json.load(f)
            all_results.extend(data)
            print(f"{len(data)} results")
    
    print(f"\nTotal loaded: {len(all_results)} results")
    
    if deduplicate:
        # Create unique keys
        seen = {}
        unique_results = []
        
        for result in all_results:
            # Convert problem_size to hashable tuple
            ps_str = json.dumps(result["problem_size"], sort_keys=True)
            key = (result["kernel"], result["backend"], ps_str)
            
            if key in seen:
                # Keep the result with better (lower) latency
                if result["latency_ms"] < seen[key]["latency_ms"]:
                    print(f"  Replacing duplicate: {result['kernel']}-{result['backend']} "
                          f"(old: {seen[key]['latency_ms']:.4f}ms, new: {result['latency_ms']:.4f}ms)")
                    # Replace
                    idx = unique_results.index(seen[key])
                    unique_results[idx] = result
                    seen[key] = result
            else:
                seen[key] = result
                unique_results.append(result)
        
        print(f"After deduplication: {len(unique_results)} results")
        final_results = unique_results
    else:
        final_results = all_results
    
    # Save
    if output:
        output_path = Path(output)
    else:
        output_path = Path("results/merged_results.json")
    
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    with open(output_path, "w") as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n✓ Merged results saved to: {output_path}")
    
    # Print statistics
    from collections import Counter
    backends = Counter(r["backend"] for r in final_results)
    kernels = Counter(r["kernel"] for r in final_results)
    
    print("\nStatistics:")
    print(f"  Kernels: {dict(kernels)}")
    print(f"  Backends: {dict(backends)}")


def main():
    parser = argparse.ArgumentParser(description="Merge benchmark result files")
    parser.add_argument("files", nargs="+", help="Input JSON files to merge")
    parser.add_argument("-o", "--output", default=None, help="Output file path")
    parser.add_argument("--no-dedup", action="store_true", help="Don't deduplicate results")
    args = parser.parse_args()
    
    merge_results(*args.files, output=args.output, deduplicate=not args.no_dedup)


if __name__ == "__main__":
    main()
