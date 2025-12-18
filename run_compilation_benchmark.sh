#!/bin/bash
# Compilation Benchmark Script
# Measures time to first kernel execution with clean cache
# Each measurement runs in a separate Python process to avoid in-memory caching

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Default parameters
OUTPUT_DIR="results/compilation"
REPEAT=5
AUTO_VISUALIZE=true

# Default configs (as JSON strings) - use small sizes for compilation benchmark
declare -A CONFIGS
CONFIGS[gemm]='{"M": 256, "N": 256, "K": 256, "dtype": "float16"}'
CONFIGS[mha]='{"batch": 1, "heads": 8, "seq_len": 512, "dim": 64}'
CONFIGS[nsa]='{"batch": 2, "heads": 16, "seq_len": 1024, "dim": 32, "block_size": 32, "groups": 16, "selected_blocks": 1, "is_causal": true, "scale": 0.1}'


# Parse arguments
BACKENDS="cuda cutedsl"
KERNELS="gemm mha nsa"

while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --backends)
            BACKENDS="$2"
            shift 2
            ;;
        --kernels)
            KERNELS="$2"
            shift 2
            ;;
        --repeat)
            REPEAT="$2"
            shift 2
            ;;
        --no-visualize)
            AUTO_VISUALIZE=false
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --output-dir DIR     Output directory (default: results/compilation)"
            echo "  --backends LIST      Space-separated backends (default: 'cuda cutedsl')"
            echo "  --kernels LIST       Space-separated kernels (default: 'gemm mha nsa')"
            echo "  --repeat N           Repetitions per config (default: 5)"
            echo "  --no-visualize       Don't auto-generate plots"
            echo "  --help               Show this help"
            echo ""
            echo "Note: Each measurement runs in a fresh Python process to avoid in-memory caching."
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Compilation Benchmark${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Configuration:"
echo "  Output: $OUTPUT_DIR"
echo "  Backends: $BACKENDS"
echo "  Kernels: $KERNELS"
echo "  Repeat: $REPEAT"
echo ""

# Check if cache directory exists
CACHE_DIR="$HOME/.tilelang"
if [ -d "$CACHE_DIR" ]; then
    echo -e "${YELLOW}Warning: TileLang cache exists at $CACHE_DIR${NC}"
    echo -e "${YELLOW}This benchmark will repeatedly clear this cache!${NC}"
    echo ""
    read -p "Continue? (y/n): " choice
    if [ "$choice" != "y" ] && [ "$choice" != "Y" ]; then
        echo "Aborted."
        exit 0
    fi
    echo ""
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TEMP_DIR="$OUTPUT_DIR/temp_${TIMESTAMP}"
mkdir -p "$TEMP_DIR"

# Track all result files
ALL_RESULTS=()

# Count total tests
TOTAL_TESTS=0
for KERNEL in $KERNELS; do
    for BACKEND in $BACKENDS; do
        TOTAL_TESTS=$((TOTAL_TESTS + REPEAT))
    done
done

CURRENT=0

echo -e "${GREEN}Starting compilation benchmark...${NC}"
echo -e "${YELLOW}Note: Each test runs in a fresh Python process${NC}"
echo ""

# Run all combinations
for KERNEL in $KERNELS; do
    if [ -z "${CONFIGS[$KERNEL]}" ]; then
        echo -e "${YELLOW}Warning: No config for kernel '$KERNEL', skipping${NC}"
        continue
    fi
    
    CONFIG_JSON="${CONFIGS[$KERNEL]}"
    
    echo -e "${BLUE}=== Benchmarking ${KERNEL^^} Compilation ===${NC}"
    
    for BACKEND in $BACKENDS; do
        echo -e "${GREEN}Backend: $BACKEND${NC}"
        
        for ((i=1; i<=REPEAT; i++)); do
            CURRENT=$((CURRENT + 1))
            
            # Unique output file for this run
            RUN_FILE="$TEMP_DIR/${KERNEL}_${BACKEND}_run${i}.json"
            
            printf "[%d/%d] %s - %s (run %d/%d) ... " $CURRENT $TOTAL_TESTS "$KERNEL" "$BACKEND" $i $REPEAT
            
            # Run compilation benchmark in fresh Python process
            # Set environment variable only for cutedsl backend
            if [ "$BACKEND" = "cutedsl" ]; then
                # Run with TILELANG_USE_GEMM_V1=1 for CuTeDSL
                if TILELANG_USE_GEMM_V1=1 python compilation_benchmark.py \
                    --kernel "$KERNEL" \
                    --backend "$BACKEND" \
                    --config "$CONFIG_JSON" \
                    --output "$RUN_FILE" \
                    --run-id "$i" > /dev/null 2>&1 ; then
                    
                    # Extract compile time from result
                    COMPILE_TIME=$(python -c "import json; print(f\"{json.load(open('$RUN_FILE'))['compile_time']:.3f}\")")
                    echo -e "${GREEN}OK (${COMPILE_TIME}s)${NC}"
                    
                    ALL_RESULTS+=("$RUN_FILE")
                else
                    echo -e "${RED}FAILED${NC}"
                fi
            else
                # Run without environment variable for other backends
                if python compilation_benchmark.py \
                    --kernel "$KERNEL" \
                    --backend "$BACKEND" \
                    --config "$CONFIG_JSON" \
                    --output "$RUN_FILE" \
                    --run-id "$i" > /dev/null 2>&1 ; then
                    
                    # Extract compile time from result
                    COMPILE_TIME=$(python -c "import json; print(f\"{json.load(open('$RUN_FILE'))['compile_time']:.3f}\")")
                    echo -e "${GREEN}OK (${COMPILE_TIME}s)${NC}"
                    
                    ALL_RESULTS+=("$RUN_FILE")
                else
                    echo -e "${RED}FAILED${NC}"
                fi
            fi
        done
        echo ""
    done
done

echo ""
echo -e "${BLUE}========================================${NC}"
echo ""

# Aggregate results
if [ ${#ALL_RESULTS[@]} -eq 0 ]; then
    echo -e "${RED}✗ No successful results${NC}"
    exit 1
fi

echo -e "${GREEN}Aggregating ${#ALL_RESULTS[@]} measurements...${NC}"

# Aggregate with Python
FINAL_JSON="$OUTPUT_DIR/compile_results_${TIMESTAMP}.json"

python -c "
import json
import sys
from collections import defaultdict

# Load all individual results
all_data = []
for f in sys.argv[1:]:
    try:
        with open(f) as fp:
            all_data.append(json.load(fp))
    except:
        pass

# Group by kernel+backend
groups = defaultdict(list)
for item in all_data:
    key = (item['kernel'], item['backend'])
    groups[key].append(item['compile_time'])

# Compute statistics
results = []
for (kernel, backend), times in groups.items():
    # Get metadata from first item
    metadata = next(d for d in all_data if d['kernel'] == kernel and d['backend'] == backend)
    
    # Calculate average: remove min and max if we have at least 3 samples
    if len(times) >= 3:
        trimmed_times = sorted(times)[1:-1]  # Remove min (first) and max (last)
        avg_time = sum(trimmed_times) / len(trimmed_times)
    else:
        avg_time = sum(times) / len(times)
    
    results.append({
        'kernel': kernel,
        'backend': backend,
        'arch': metadata['arch'],
        'gpu_model': metadata['gpu_model'],
        'cpu_model': metadata['cpu_model'],
        'compute_capability': metadata['compute_capability'],
        'config': metadata['config'],
        'compile_time_avg': avg_time,
        'compile_time_min': min(times),
        'compile_time_max': max(times),
        'samples': times
    })

# Save
with open('$FINAL_JSON', 'w') as f:
    json.dump(results, f, indent=2)

print(f'✓ Aggregated results saved to: $FINAL_JSON')
" "${ALL_RESULTS[@]}"

# Clean up temp files
rm -rf "$TEMP_DIR"

echo ""
echo -e "${GREEN}✓ Compilation benchmark complete!${NC}"
echo ""

# Visualize
if [ "$AUTO_VISUALIZE" = true ]; then
    echo -e "${GREEN}Generating visualizations...${NC}"
    echo ""
    
    if python visualize_compilation.py "$FINAL_JSON" --output "$OUTPUT_DIR"; then
        echo ""
        echo -e "${GREEN}✓ Visualization complete!${NC}"
        echo ""
        echo "Generated plots:"
        ls -1 $OUTPUT_DIR/*.png 2>/dev/null | tail -5 | sed 's/^/  /'
    else
        echo -e "${RED}✗ Visualization failed${NC}"
    fi
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}✓ All done!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Results: $FINAL_JSON"
echo ""
