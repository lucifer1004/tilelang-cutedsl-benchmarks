#!/bin/bash
# CuTeDSL Benchmark Automation Script
# Handles separate backend execution and result merging

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default parameters
CONFIG="configs/default.json"
WARMUP=5
ITERS=10
OUTPUT_DIR="results"
RUN_BASELINES=true
RUN_CUTEDSL=true
AUTO_MERGE=true
AUTO_VISUALIZE=true
VERIFY=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --warmup)
            WARMUP="$2"
            shift 2
            ;;
        --iters)
            ITERS="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --skip-baselines)
            RUN_BASELINES=false
            shift
            ;;
        --skip-cutedsl)
            RUN_CUTEDSL=false
            shift
            ;;
        --no-merge)
            AUTO_MERGE=false
            shift
            ;;
        --no-visualize)
            AUTO_VISUALIZE=false
            shift
            ;;
        --verify)
            VERIFY=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --config FILE         Config file (default: configs/default.json)"
            echo "  --warmup N           Warmup iterations (default: 10)"
            echo "  --iters N            Benchmark iterations (default: 100)"
            echo "  --output-dir DIR     Output directory (default: results)"
            echo "  --skip-baselines     Skip CUDA/PyTorch baselines"
            echo "  --skip-cutedsl       Skip CuTeDSL backend"
            echo "  --no-merge           Don't auto-merge results"
            echo "  --no-visualize       Don't auto-generate plots"
            echo "  --verify             Verify correctness (slower)"
            echo "  --help               Show this help"
            echo ""
            echo "Note: PyTorch NSA is skipped by default - too slow, only use for correctness check."
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Create timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  CuTeDSL Benchmark Runner${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Configuration:"
echo "  Config file: $CONFIG"
echo "  Warmup: $WARMUP, Iterations: $ITERS"
echo "  Output dir: $OUTPUT_DIR"
echo "  Verify: $VERIFY"
echo ""

# File paths
BASELINE_FILE="$OUTPUT_DIR/baselines_${TIMESTAMP}.json"
CUTEDSL_FILE="$OUTPUT_DIR/cutedsl_${TIMESTAMP}.json"
MERGED_FILE="$OUTPUT_DIR/merged_${TIMESTAMP}.json"

# Track which files were created
CREATED_FILES=()

# Step 1: Run baselines (CUDA + PyTorch for GEMM/MHA, CUDA only for NSA)
if [ "$RUN_BASELINES" = true ]; then
    echo -e "${GREEN}[Step 1a/4] Running GEMM/MHA baselines (CUDA + PyTorch)...${NC}"
    echo ""
    
    GEMM_MHA_FILE="$OUTPUT_DIR/gemm_mha_baselines_${TIMESTAMP}.json"
    BASELINE_CMD="python runner.py --config $CONFIG --warmup $WARMUP --iters $ITERS --kernels gemm mha --backends cuda pytorch --output $GEMM_MHA_FILE"
    if [ "$VERIFY" = true ]; then
        BASELINE_CMD="$BASELINE_CMD --verify"
    fi
    
    if eval "$BASELINE_CMD"; then
        echo ""
        echo -e "${GREEN}✓ GEMM/MHA baselines complete${NC}"
        CREATED_FILES+=("$GEMM_MHA_FILE")
    else
        echo -e "${RED}✗ GEMM/MHA baseline benchmark failed${NC}"
        exit 1
    fi
    
    echo ""
    echo -e "${GREEN}[Step 1b/4] Running NSA baseline (CUDA only, PyTorch too slow)...${NC}"
    echo ""
    
    NSA_FILE="$OUTPUT_DIR/nsa_baseline_${TIMESTAMP}.json"
    NSA_CMD="python runner.py --config $CONFIG --warmup $WARMUP --iters $ITERS --kernels nsa --backends cuda --output $NSA_FILE"
    if [ "$VERIFY" = true ]; then
        NSA_CMD="$NSA_CMD --verify"
    fi
    
    if eval "$NSA_CMD"; then
        echo ""
        echo -e "${GREEN}✓ NSA baseline complete${NC}"
        CREATED_FILES+=("$NSA_FILE")
    else
        echo -e "${RED}✗ NSA baseline benchmark failed${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}[Step 1/4] Skipping baselines${NC}"
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo ""

# Step 2: Run CuTeDSL (with automatic environment setup)
if [ "$RUN_CUTEDSL" = true ]; then
    echo -e "${GREEN}[Step 2/4] Running CuTeDSL backend...${NC}"
    echo -e "${BLUE}(Running with TILELANG_USE_GEMM_V1=1)${NC}"
    echo ""
    
    CUTEDSL_CMD="TILELANG_USE_GEMM_V1=1 python runner.py --config $CONFIG --warmup $WARMUP --iters $ITERS --backends cutedsl --output $CUTEDSL_FILE"
    if [ "$VERIFY" = true ]; then
        CUTEDSL_CMD="$CUTEDSL_CMD --verify"
    fi
    
    if eval "$CUTEDSL_CMD"; then
        echo ""
        echo -e "${GREEN}✓ CuTeDSL complete${NC}"
        CREATED_FILES+=("$CUTEDSL_FILE")
    else
        echo ""
        echo -e "${RED}✗ CuTeDSL benchmark failed${NC}"
        echo -e "${YELLOW}Note: TILELANG_USE_GEMM_V1 was set to 1${NC}"
        echo -e "${YELLOW}Continuing with available results...${NC}"
    fi
else
    echo -e "${YELLOW}[Step 2/4] Skipping CuTeDSL${NC}"
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo ""

# Step 3: Merge results
if [ "$AUTO_MERGE" = true ] && [ ${#CREATED_FILES[@]} -gt 0 ]; then
    echo -e "${GREEN}[Step 3/4] Merging results...${NC}"
    echo ""
    
    if [ ${#CREATED_FILES[@]} -eq 1 ]; then
        echo "Only one result file, copying to merged output"
        cp "${CREATED_FILES[0]}" "$MERGED_FILE"
        echo -e "${GREEN}✓ Results saved to $MERGED_FILE${NC}"
    else
        if python merge_results.py "${CREATED_FILES[@]}" -o "$MERGED_FILE"; then
            echo ""
            echo -e "${GREEN}✓ Results merged to $MERGED_FILE${NC}"
        else
            echo -e "${RED}✗ Merge failed${NC}"
            MERGED_FILE="${CREATED_FILES[0]}"
            echo -e "${YELLOW}Using first result file: $MERGED_FILE${NC}"
        fi
    fi
else
    echo -e "${YELLOW}[Step 3/4] Skipping merge${NC}"
    if [ ${#CREATED_FILES[@]} -gt 0 ]; then
        MERGED_FILE="${CREATED_FILES[0]}"
    fi
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo ""

# Step 4: Visualize (optional)
if [ "$AUTO_VISUALIZE" = true ] && [ -f "$MERGED_FILE" ]; then
    echo -e "${GREEN}[Step 4/4] Generating visualizations...${NC}"
    echo ""
    
    if python visualize.py "$MERGED_FILE"; then
        echo ""
        echo -e "${GREEN}✓ Plots generated in $OUTPUT_DIR/${NC}"
    else
        echo -e "${RED}✗ Visualization failed${NC}"
    fi
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}✓ Benchmark pipeline complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Results:"
for file in "${CREATED_FILES[@]}"; do
    echo "  - $file"
done
if [ -f "$MERGED_FILE" ] && [ ${#CREATED_FILES[@]} -gt 1 ]; then
    echo "  - $MERGED_FILE (merged)"
fi
echo ""
echo "Generated plots in: $OUTPUT_DIR/"
echo ""
echo "To visualize again:"
echo "  python visualize.py $MERGED_FILE"
echo ""
