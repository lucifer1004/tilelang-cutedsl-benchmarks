# Quick Start Guide

## Basic Usage

### 1. Run Full Benchmark Suite

```bash
# Run with default config (all backends, default problem sizes)
# Default: warmup=5, iters=10
python runner.py

# For production benchmarking (more accurate)
python runner.py --warmup 20 --iters 200

# This will test:
# - GEMM: 1024x1024x1024, 4096x4096x4096, 8192x8192x8192
# - MHA: B8_H32_S1024_D128, B8_H32_S4096_D128, B8_H128_S4096_D128
# - NSA: Various configurations (see configs/default.json)
# - Backends: cutedsl, cuda, pytorch
```

### 2. Run Specific Configurations

```bash
# Only GEMM kernel
python runner.py --kernels gemm

# Only CUDA and PyTorch baselines (skip CuTeDSL)
python runner.py --backends cuda pytorch

# More iterations for accurate results
python runner.py --warmup 20 --iters 200

# Quick test with minimal iterations (default is already fast: warmup=5, iters=10)
python runner.py --warmup 2 --iters 5
```

### 3. Visualize Results

```bash
# Visualize the latest result (note: filename includes GPU model)
python visualize.py results/sm_90_NVIDIA_H200_*.json

# Or specify exact file
python visualize.py results/sm_90_NVIDIA_H200_20251218_143052.json
```

This generates PNG plots in the `results/` directory.

## Expected Output

### Runner Output

```
Detected GPU architecture: sm_90
Configuration: warmup=5, iters=10

=== Benchmarking GEMM ===
[1/18]   [gemm] cutedsl {'M': 1024, ...} ... OK (0.0123 ms, 175.23 TFLOPS)
[2/18]   [gemm] cuda {'M': 1024, ...} ... OK (0.0175 ms, 122.51 TFLOPS)
[3/18]   [gemm] pytorch {'M': 1024, ...} ... OK (0.0114 ms, 187.87 TFLOPS)
...

✓ Benchmark complete!
  Total: 18/18 successful
  Results saved to: results/sm_90_20251217_191248.json
```

### Visualize Output

Performance benchmarks:
- `{kernel}_comparison_{GPU_MODEL}.png` - Side-by-side latency and throughput comparison
  - Example: `gemm_comparison_NVIDIA_H200.png`
- `speedup_comparison_{GPU_MODEL}.png` - Speedup vs baseline (PyTorch for GEMM/MHA, CUDA for NSA)

## Customizing Test Cases

Three configuration files are available:

### 1. `configs/default.json` (alias: `configs/full.json`)
Full benchmark with multiple problem sizes for all kernels.

### 2. `configs/test.json`
Quick validation with one problem size per kernel. Good for correctness checks.

```bash
python runner.py --config configs/test.json --verify
```

### Creating Custom Configs

Create your own JSON config file:

```json
{
  "gemm": [
    {"M": 2048, "N": 2048, "K": 2048, "dtype": "float16"},
    {"M": 16384, "N": 16384, "K": 16384, "dtype": "float16"}
  ],
  "mha": [
    {"batch": 16, "heads": 64, "seq_len": 2048, "dim": 64}
  ],
  "nsa": [
    {"batch": 2, "heads": 16, "seq_len": 1024, "dim": 64, "block_size": 32, "groups": 1, "selected_blocks": 16, "is_causal": true}
  ]
}
```

Then run:

```bash
python runner.py --config my_custom_config.json
```

## Environment Setup for CuTeDSL

If you see:

```
FAILED (ValueError: CuTeDSL backend requires GEMM v1. Please set envir)
```

You need to set environment variables before running. Check your TileLang installation docs.

## Troubleshooting

### Issue: ImportError

```bash
# Make sure you're in the right directory
cd tilelang-cutedsl-benchmarks

# Check Python can find tilelang
python -c "import tilelang; print(tilelang.__file__)"
```

### Issue: CUDA out of memory

Reduce problem sizes in config or test one kernel at a time:

```bash
python runner.py --kernels gemm --backends cuda
```

### Issue: Visualization fails

```bash
# Install dependencies
pip install -r requirements.txt

# Check matplotlib backend
python -c "import matplotlib; print(matplotlib.get_backend())"
```

## Cross-Architecture Workflow

```bash
# On Ampere machine (SM 80)
ssh ampere-node
cd tilelang-cutedsl-benchmarks
python runner.py --backends cuda pytorch
# Result: results/sm_80_*.json

# On Hopper machine (SM 90)
ssh hopper-node
cd tilelang-cutedsl-benchmarks
python runner.py --backends cuda pytorch
# Result: results/sm_90_*.json

# Compare results
python visualize.py results/sm_80_*.json
python visualize.py results/sm_90_*.json
```

## Result Format

JSON structure for programmatic analysis:

```python
import json
import pandas as pd

data = json.load(open('results/sm_90_*.json'))
df = pd.DataFrame(data)

# Find best backend for each problem size
best = df.loc[df.groupby(['kernel', 'problem_size'])['tflops'].idxmax()]
print(best[['kernel', 'backend', 'tflops']])
```
