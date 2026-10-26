#!/usr/bin/env bash
# Launch one worker batch in the site runtime environment.
# Args: <gpu> <config_dir> <band> <batch> <manifest|STITCH>
set -eo pipefail
GPU=$1; CONFIG_DIR=$2; BAND=$3; BATCH=$4; MANIFEST=$5

# Pin this launcher too (its tee and the conda helpers), not only the worker.
if [[ -n ${GSI_TASKSET_CPUS:-} ]]; then
    taskset -cp "$GSI_TASKSET_CPUS" $$ >/dev/null
fi

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/.." && pwd)

CONDA_SH=${GSI_CONDA_SH:-/opt/miniconda3/etc/profile.d/conda.sh}
GSI_RUNTIME_ENV=${GSI_RUNTIME_ENV:-/opt/devel/pipeline/envs/py38_orca_nkosogor}

source "$CONDA_SH"
conda activate "$GSI_RUNTIME_ENV"

mapfile -t CONFIG_ENV < <(python - "$CONFIG_DIR" <<'PY'
import sys
from pathlib import Path
import yaml

env = yaml.safe_load((Path(sys.argv[1]) / "pipeline.yaml").read_text())["env"]
for key in ("dev_env", "julia_bin", "shared_depot", "ttcalx"):
    print(env[key])
PY
)
GSI_DEV_ENV=${GSI_DEV_ENV:-${CONFIG_ENV[0]}}
GSI_JULIA_BIN=${GSI_JULIA_BIN:-${CONFIG_ENV[1]}}
GSI_SHARED_DEPOT=${GSI_SHARED_DEPOT:-${CONFIG_ENV[2]}}
GSI_TTCALX_DIR=${GSI_TTCALX_DIR:-${CONFIG_ENV[3]}}

export OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=$GPU
export JULIA_CUDA_USE_COMPAT=false
export JULIA_CUDA_NONBLOCKING_SYNCHRONIZATION=false
export JULIA_DEPOT_PATH="/tmp/gsi_depot_${USER}:$GSI_SHARED_DEPOT"
export PATH="$GSI_JULIA_BIN:/opt/bin:$PATH"
export TTCALX_DIR="$GSI_TTCALX_DIR"
export FPACK_BIN="$GSI_DEV_ENV/bin/fpack"
export GSI_WORKERS="$HERE"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"

# Keep worker logs beside the date's products.
LOG_ROOT=$(python - "$CONFIG_DIR" "$BAND" <<'PY'
import sys
from pathlib import Path
import yaml

config_dir, band = sys.argv[1], sys.argv[2]
try:
    pipeline = yaml.safe_load((Path(config_dir) / "pipeline.yaml").read_text())
    if pipeline.get("layout") == "lst":
        root = (Path(pipeline["output_root"]) / "_logs" / str(pipeline["date"]) /
                str(pipeline["run_label"]) / f"{band}MHz")
    else:
        root = Path(pipeline["output_root"]) / f"{band}MHz" / str(pipeline["date"]) / "logs"
except Exception:
    root = Path(config_dir) / "logs"
print(root)
PY
)
mkdir -p "$LOG_ROOT"
LOG="$LOG_ROOT/worker-${BAND}_${BATCH}-$(hostname -s)-gpu${GPU}.log"
if [[ $MANIFEST == HOUR_IMAGE:* ]]; then
    LOG="$LOG_ROOT/hour_image-${BAND}_${MANIFEST#HOUR_IMAGE:}-$(hostname -s)-gpu${GPU}.log"
fi
# Optional CPU pinning / priority (inherited by wsclean, Julia, casa children):
# GSI_TASKSET_CPUS=0-23 GSI_NICE=10 GSI_IONICE="-c2 -n7".
WRAP=()
[[ -n ${GSI_TASKSET_CPUS:-} ]] && WRAP+=(taskset -c "$GSI_TASKSET_CPUS")
[[ -n ${GSI_NICE:-} ]] && WRAP+=(nice -n "$GSI_NICE")
[[ -n ${GSI_IONICE:-} ]] && WRAP+=(ionice $GSI_IONICE)
"${WRAP[@]}" python -m gpu_subband_imaging.worker \
    "$CONFIG_DIR" "$BAND" "$BATCH" "$MANIFEST" --gpu "$GPU" 2>&1 | tee "$LOG"
