#!/usr/bin/env bash
# Geometric (precessed) UVW sky model vs CPU TTCal and the data; reuses a
# test_genvis.sh output directory (needs cpu_sky.ms there).
#   GPU=0 test_geom.sh PREP_MS OUT_DIR
set -eo pipefail
PREP=$1; OUT=$2
source "$(dirname "$0")/env.sh"
export CUDA_VISIBLE_DEVICES=${GPU:-0}
cd "$OUT"
for n in gpu_sky_geom gpu_sky_geompos; do rm -rf $n.ms; cp -r "$PREP" $n.ms; done
J="$JULIA_BIN/julia"
gv() { TTCALX_DIR=$TTCALX_EXP $J --project=$TTCALX_EXP "$PX_SCRIPTS/genvis_gpu.jl" "$@" 2>&1 | grep "GENVIS\|UVW_\|ERROR"; }
gv $PWD/gpu_sky_geom.ms "$SKY_MODEL" geom
gv $PWD/gpu_sky_geompos.ms "$SKY_MODEL" geom_pos
echo "== sky models vs CPU TTCal (julia060)"
px_py "$PX_SCRIPTS/compare_models.py" cpu_sky.ms gpu_sky_auto.ms gpu_sky_geom.ms gpu_sky_geompos.ms
echo "== raw model vs data"
px_py "$PX_SCRIPTS/model_data_corr.py" "$PREP" cpu_sky.ms gpu_sky_auto.ms gpu_sky_geom.ms gpu_sky_geompos.ms
