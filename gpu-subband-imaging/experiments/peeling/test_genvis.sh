#!/usr/bin/env bash
# CPU vs GPU model visibilities for the sky and RFI models on one prep MS,
# and how well each raw model matches the data.
#   GPU=0 test_genvis.sh PREP_MS OUT_DIR
set -eo pipefail
PREP=$1; OUT=$2
source "$(dirname "$0")/env.sh"
export CUDA_VISIBLE_DEVICES=${GPU:-0}
# CPU TTCal (ttcal_dev) reads the RFI WGS84 entry as Cartesian; this is its ITRF result.
CPU_RFI_ITRF="6.368747415744499e6,-364625.3282460041,114589.56805339623"
mkdir -p "$OUT"; cd "$OUT"
V="cpu_sky gpu_sky_base gpu_sky_exp gpu_sky_auto cpu_rfi gpu_rfi_cpupos gpu_rfi gpu_rfi_nogeom"
for n in $V; do rm -rf $n.ms; cp -r "$PREP" $n.ms; done
env LD_LIBRARY_PATH=/opt/astro/mwe/usr/lib64:/opt/astro/lib/ AIPSPATH='/opt/astro/casa-data dummy dummy' \
    /opt/devel/pipeline/envs/julia060/bin/julia "$PX_SCRIPTS/genvis_cpu.jl" $PWD/cpu_sky.ms "$SKY_MODEL"
( source /opt/miniconda3/etc/profile.d/conda.sh; conda activate ttcal_dev
  LD_LIBRARY_PATH="/opt/lib:${LD_LIBRARY_PATH:-}" julia "$PX_SCRIPTS/genvis_cpu.jl" $PWD/cpu_rfi.ms "$RFI_MODEL" )
J="$JULIA_BIN/julia"
gv() { local dir=$1; shift; TTCALX_DIR=$dir $J --project=$dir "$PX_SCRIPTS/genvis_gpu.jl" "$@" 2>&1 | grep "GENVIS\|UVW_\|ERROR"; }
gv $TTCALX_BASE $PWD/gpu_sky_base.ms "$SKY_MODEL"
gv $TTCALX_EXP  $PWD/gpu_sky_exp.ms  "$SKY_MODEL" as_is
gv $TTCALX_EXP  $PWD/gpu_sky_auto.ms "$SKY_MODEL" auto
gv $TTCALX_EXP  $PWD/gpu_rfi_cpupos.ms "$RFI_MODEL" auto uvw "$CPU_RFI_ITRF"
gv $TTCALX_EXP  $PWD/gpu_rfi.ms        "$RFI_MODEL" auto uvw
gv $TTCALX_EXP  $PWD/gpu_rfi_nogeom.ms "$RFI_MODEL" auto none
echo "== sky models vs CPU TTCal (julia060)"
px_py "$PX_SCRIPTS/compare_models.py" cpu_sky.ms gpu_sky_base.ms gpu_sky_exp.ms gpu_sky_auto.ms
echo "== rfi models vs CPU TTCal (ttcal_dev)"
px_py "$PX_SCRIPTS/compare_models.py" cpu_rfi.ms gpu_rfi_cpupos.ms gpu_rfi.ms
echo "== raw model vs data"
px_py "$PX_SCRIPTS/model_data_corr.py" "$PREP" $(for n in $V; do echo $n.ms; done)
