#!/bin/bash
# Quick script to run only CuTeDSL backend
# Useful when you've already run baselines

set -e

echo "=========================================="
echo "  CuTeDSL Backend Only"
echo "=========================================="
echo ""
echo "This will run ONLY the CuTeDSL backend."
echo ""

# Default parameters
CONFIG="${1:-configs/default.json}"
OUTPUT="${2:-results/cutedsl_$(date +%Y%m%d_%H%M%S).json}"

echo "Configuration:"
echo "  Config: $CONFIG"
echo "  Output: $OUTPUT"
echo "  Environment: TILELANG_USE_GEMM_V1=1"
echo ""

TILELANG_USE_GEMM_V1=1 python runner.py --config "$CONFIG" --backends cutedsl --output "$OUTPUT"

echo ""
echo "✓ CuTeDSL benchmark complete!"
echo "  Results: $OUTPUT"
echo ""
echo "To merge with baselines:"
echo "  python merge_results.py results/baselines_*.json $OUTPUT -o results/merged.json"
echo ""
