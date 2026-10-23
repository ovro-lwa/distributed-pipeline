# Common environment for the peeling experiments (source, don't execute).
# Everything lives under $PX; production install paths are only read.
export PX=${PX:-/lustre/nkosogor/peel_exp}
export PX_SCRIPTS=$PX/repo/experiments/peeling
export PX_REPO=$PX/repo                       # rsync of gpu-subband-imaging (exp branch)
export TTCALX_BASE=$PX/ttcalx_base            # copy of deployed TTCalX (main @ 62f8c54)
export TTCALX_EXP=$PX/ttcalx_exp              # rsync of TTCalX exp/peeling-experiments

# Read-only production references.
export SKY_MODEL=/lustre/gh/calibration/pipeline/reference/sources/sources.json
export RFI_MODEL=/lustre/gh/calibration/pipeline/reference/sources/rfi_43.2_ver20251101.json
export XY_TABLE=/lustre/gh/polcal/xyphase_delay_pos_3.8643ns.Xf
export JULIA_BIN=/opt/devel/nkosogor/nkosogor/julia-1.10.4/bin
export SHARED_DEPOT=/opt/devel/nkosogor/nkosogor/julia_depot
export PY_ENV=/opt/devel/pipeline/envs/py38_orca_nkosogor
export DEV_PY=/opt/devel/pipeline/envs/development/bin/python
export WSCLEAN=/opt/bin/wsclean

# Our own depot first so nothing is ever compiled into the shared one.
export JULIA_DEPOT_PATH="$PX/depot:$SHARED_DEPOT"
export JULIA_CUDA_USE_COMPAT=false
export JULIA_CUDA_NONBLOCKING_SYNCHRONIZATION=false
export OPENBLAS_NUM_THREADS=1

px_py() {
    source /opt/miniconda3/etc/profile.d/conda.sh
    conda activate "$PY_ENV"
    PYTHONPATH="$PX_REPO/src:$PX_SCRIPTS:${PYTHONPATH:-}" python "$@"
}
