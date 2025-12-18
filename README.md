# TileLang CuTeDSL Performance Benchmarks

Systematic CuTeDSL backend performance testing framework, comparing CuTeDSL, TileLang CUDA, and PyTorch backends.

## Quick Start

### Method 1: Automated Script (Recommended)

```bash
# Run complete benchmark workflow with one command
./run_benchmark.sh
```

The script will automatically:
1. Run CUDA and PyTorch baseline tests
2. Set TILELANG_USE_GEMM_V1=1 and run CuTeDSL
3. Merge all results
4. Generate visualization charts

### Method 2: Manual Step-by-Step Execution

```bash
# Step 1: Run baseline tests
python runner.py --backends cuda pytorch --output results/baselines.json

# Step 2: Run CuTeDSL (environment variables set automatically)
python runner.py --backends cutedsl --output results/cutedsl.json

# Step 3: Merge results
python merge_results.py results/baselines.json results/cutedsl.json -o results/complete.json

# Step 4: Visualize
python visualize.py results/complete.json
```

## Supported Kernels

### 1. GEMM (Matrix Multiplication)
- Metrics: Latency + TFLOPS
- Configuration: M, N, K, dtype

### 2. MHA (Multi-Head Attention)
- Metrics: Latency + TFLOPS
- Configuration: batch, heads, seq_len, dim

### 3. NSA (Native Sparse Attention) ⭐New
- Metrics: Latency + TFLOPS
- Configuration: batch, heads, seq_len, dim, block_size, groups, selected_blocks, is_causal
- Based on DeepSeek NSA implementation

## Core Features

### 1. Complete GPU Information
Automatically captures full GPU information, not just SM architecture:
```json
{
  "arch": "sm_90",
  "gpu_model": "NVIDIA H100 PCIe",
  "compute_capability": "9.0"
}
```

### 2. Separate Backend Execution
Supports running backends separately, then merging:
- CuTeDSL automatically sets environment variables (TILELANG_USE_GEMM_V1=1)
- Baseline tests can run independently first
- Intelligent deduplication and result merging

### 3. Flexible Configuration

#### Full Test (default.json)
```bash
python runner.py --config configs/default.json
```

#### Quick Validation (test.json)
```bash
python runner.py --config configs/test.json --warmup 2 --iters 5
```

#### NSA-Specific Test (nsa_only.json)
```bash
python runner.py --config configs/nsa_only.json --kernels nsa
```

## Configuration File Details

### configs/default.json
Complete benchmark configuration, including multiple test cases for all kernels.

### configs/test.json
Quick test configuration, tests only 1 size per kernel for verification.

### configs/nsa_only.json
NSA-specific test configuration with different parameter combinations:
- Different sequence lengths: 256, 512, 1024, 2048
- Different block_size: 32, 64
- Different groups configurations: 1, 4 (for GQA)
- Different sparsity: varying selected_blocks

## Script Tools

### `run_benchmark.sh` - Fully Automated Workflow

```bash
# Basic usage
./run_benchmark.sh

# Custom parameters
./run_benchmark.sh --config configs/nsa_only.json --warmup 20 --iters 200

# Skip certain steps
./run_benchmark.sh --skip-cutedsl           # Run baselines only
./run_benchmark.sh --skip-baselines         # Run CuTeDSL only

# Test specific kernel
./run_benchmark.sh --kernels nsa

# View all options
./run_benchmark.sh --help
```

### `run_cutedsl_only.sh` - Quick CuTeDSL Run

```bash
# Use default configuration
./run_cutedsl_only.sh

# Specify configuration and output
./run_cutedsl_only.sh configs/nsa_only.json results/my_cutedsl.json
```

### `merge_results.py` - Merge Multiple Results

```bash
# Merge two files
python merge_results.py results/file1.json results/file2.json -o results/merged.json

# Merge results from multiple GPUs
python merge_results.py results/h100_*.json results/a100_*.json -o results/multi_gpu.json
```

## NSA Configuration Example

```json
{
  "nsa": [
    {
      "batch": 2,
      "heads": 16,
      "seq_len": 512,
      "dim": 64,
      "block_size": 32,
      "groups": 1,
      "selected_blocks": 8,
      "is_causal": true,
      "scale": 0.1
    }
  ]
}
```

Parameter descriptions:
- `batch`: batch size
- `heads`: total number of attention heads (for GQA, this is the number of query heads)
- `seq_len`: sequence length
- `dim`: head dimension
- `block_size`: sparse block size
- `groups`: number of GQA groups (heads should be divisible by groups)
- `selected_blocks`: number of sparse blocks selected per token
- `is_causal`: whether to use causal mask
- `scale`: attention scale (optional, defaults to 1/sqrt(dim))

## Common Use Cases

### Use Case 1: First Complete Test

```bash
./run_benchmark.sh
```

### Use Case 2: Test NSA Only

```bash
python runner.py --config configs/nsa_only.json --kernels nsa --backends cuda pytorch
```

### Use Case 3: Cross-GPU Comparison

```bash
# On H100
ssh h100-node
./run_benchmark.sh --output-dir results/h100

# On A100
ssh a100-node  
./run_benchmark.sh --output-dir results/a100

# Merge and compare
python merge_results.py results/h100/merged_*.json results/a100/merged_*.json -o results/h100_vs_a100.json
python visualize.py results/h100_vs_a100.json
```

### Use Case 4: Quick Correctness Verification

```bash
python runner.py --config configs/test.json --verify --warmup 2 --iters 5
```

## Output Examples

### Terminal Output
```
==========================================
  CuTeDSL Benchmark Runner
==========================================

GPU Information:
  Model: NVIDIA H100 PCIe
  Architecture: sm_90 (Compute 9.0)

[Step 1/3] Running CUDA and PyTorch baselines...

=== Benchmarking NSA ===
[1/6]   [nsa] cuda {batch:2, seq_len:512...} ... OK (0.543 ms, 45.23 TFLOPS)
[2/6]   [nsa] pytorch {batch:2, seq_len:512...} ... OK (1.234 ms, 19.87 TFLOPS)

✓ Baselines complete
```

### Generated Charts
- `gemm_comparison_{GPU}.png` - GEMM latency and throughput comparison
- `mha_comparison_{GPU}.png` - MHA latency and throughput comparison  
- `nsa_comparison_{GPU}.png` - NSA latency and throughput comparison (new)
- `speedup_comparison_{GPU}.png` - Speedup relative to PyTorch

### JSON Result Format
```json
[
  {
    "kernel": "nsa",
    "backend": "cutedsl",
    "arch": "sm_90",
    "gpu_model": "NVIDIA H100 PCIe",
    "compute_capability": "9.0",
    "problem_size": {
      "batch": 2,
      "heads": 16,
      "seq_len": 512,
      "dim": 64,
      "block_size": 32,
      "groups": 1,
      "selected_blocks": 8,
      "is_causal": true
    },
    "latency_ms": 0.543,
    "tflops": 45.23,
    "compilation_time": 2.15
  }
]
```

## Dependency Installation

```bash
pip install -r requirements.txt
```

Requirements:
- matplotlib >= 3.5.0
- pandas >= 1.5.0
- torch (already included)
- tilelang (already included)
- cuda-python (already included)

## Additional Documentation

- [USAGE.md](USAGE.md) - Detailed usage guide and troubleshooting
- [WORKFLOW.md](WORKFLOW.md) - Complete workflow for separate backend execution
- [QUICKSTART.md](QUICKSTART.md) - 5-minute quick start

## License

Same as TileLang project.
