#!/usr/bin/env bash
# Precompile TTCalX and probe each GPU on one worker node.
# GSI_WARM_CUPY=1: also pre-warm the imager's CuPy kernel cache (deploy/warm_cupy_cache.py, synthetic data) with
# GSI_IMAGER_PYTHON (the deploy's venv python) into GSI_CUPY_CACHE_DIR (the config's hour_imaging.env
# CUPY_CACHE_DIR), once per GPU model of the node.
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

if [ "${GSI_WARM_CUPY:-0}" = "1" ]; then
    : "${GSI_IMAGER_PYTHON:?GSI_WARM_CUPY=1 needs GSI_IMAGER_PYTHON (e.g. /opt/devel/pipeline/gsi-v5/venv/bin/python)}"
    : "${GSI_CUPY_CACHE_DIR:?GSI_WARM_CUPY=1 needs GSI_CUPY_CACHE_DIR (the hour_imaging.env CUPY_CACHE_DIR of the config)}"
    HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
    WARMED="|"
    for g in $(seq 0 $((NGPU-1))); do
        grep -qx "gpu $g OK" "$CAPS" || continue
        model=$(nvidia-smi -i "$g" --query-gpu=name --format=csv,noheader 2>/dev/null)
        case "$WARMED" in *"|$model|"*) continue ;; esac
        echo "[$(hostname)] warming the CuPy kernel cache on gpu $g ($model) -> $GSI_CUPY_CACHE_DIR"
        (cd /tmp && CUDA_VISIBLE_DEVICES=$g CUPY_CACHE_DIR="$GSI_CUPY_CACHE_DIR" \
            ${GSI_TASKSET_CPUS:+taskset -c $GSI_TASKSET_CPUS} nice -n "${GSI_NICE:-10}" \
            "$GSI_IMAGER_PYTHON" "$HERE/warm_cupy_cache.py" ${GSI_WARM_CUPY_ARGS:-}) && WARMED="$WARMED$model|"
    done
fi
