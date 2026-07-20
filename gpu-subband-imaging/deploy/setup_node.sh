#!/usr/bin/env bash
# Precompile TTCalX and probe each GPU on one worker node.
set -eo pipefail
CONDA_SH=${GSI_CONDA_SH:-/opt/miniconda3/etc/profile.d/conda.sh}
RUNTIME_ENV=${GSI_RUNTIME_ENV:-/opt/gsi/envs/runtime}
JULIA_BIN=${GSI_JULIA_BIN:-/opt/julia/bin}
SHARED_DEPOT=${GSI_SHARED_DEPOT:-/opt/gsi/julia-depot}
TTCALX=${GSI_TTCALX_DIR:-/opt/TTCalX}
CAPS=/tmp/gsi_caps_${USER}.txt

source "$CONDA_SH"
conda activate "$RUNTIME_ENV"
export JULIA_CUDA_USE_COMPAT=false
export JULIA_DEPOT_PATH="/tmp/gsi_depot_${USER}:$SHARED_DEPOT"
export PATH="$JULIA_BIN:$PATH"

echo "[$(hostname)] precompiling Julia depot (one-time)..."
julia --project="$TTCALX" -e 'using Pkg; Pkg.precompile()' >/dev/null 2>&1 || \
julia --project="$TTCALX" -e 'using Pkg; Pkg.precompile()' >/dev/null 2>&1 || true

echo "[$(hostname)] probing GPUs..."
: > "$CAPS"
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)
for g in $(seq 0 $((NGPU-1))); do
    if CUDA_VISIBLE_DEVICES=$g timeout 120 julia --project="$TTCALX" -e '
        using CUDA; a = CUDA.rand(Float64, 512, 512); exit(sum(a*a) > 0 ? 0 : 1)
    ' >/dev/null 2>&1; then
        echo "gpu $g OK" | tee -a "$CAPS"
    else
        echo "gpu $g FAIL" | tee -a "$CAPS"
    fi
done
echo "[$(hostname)] capabilities -> $CAPS"
