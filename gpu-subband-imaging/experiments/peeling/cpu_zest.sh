#!/usr/bin/env bash
# Run the main (Celery) pipeline's CPU TTCal zest on one MS, exactly as
# orca/tasks/subband_tasks.py does (zest_with_ttcal / _peel_legacy).
#   cpu_zest.sh MS sky|rfi [MAXITER] [SOURCES]
# Peels DATA in place (TTCal uses CORRECTED_DATA only when it exists).
set -eo pipefail
MS=$1; KIND=$2; MAXITER=${3:-5}
source "$(dirname "$0")/env.sh"
ARGS="--beam constant --minuvw 5 --maxiter $MAXITER --tolerance 1e-4"
t0=$(date +%s.%N)
if [ "$KIND" = sky ]; then
    SRC=${4:-$SKY_MODEL}
    env LD_LIBRARY_PATH=/opt/astro/mwe/usr/lib64:/opt/astro/lib/ \
        AIPSPATH='/opt/astro/casa-data dummy dummy' \
        /opt/devel/pipeline/envs/julia060/bin/julia \
        /opt/devel/pipeline/envs/julia060/bin/ttcal.jl zest "$MS" "$SRC" $ARGS
elif [ "$KIND" = rfi ]; then
    SRC=${4:-$RFI_MODEL}
    source /opt/miniconda3/etc/profile.d/conda.sh
    conda activate ttcal_dev
    OMP_NUM_THREADS=8 LD_LIBRARY_PATH="/opt/lib:${LD_LIBRARY_PATH:-}" \
        ttcal.jl zest "$MS" "$SRC" $ARGS
else
    echo "kind must be sky or rfi" >&2; exit 2
fi
echo "CPU_ZEST kind=$KIND ms=$MS elapsed=$(echo "$(date +%s.%N) - $t0" | bc)"
