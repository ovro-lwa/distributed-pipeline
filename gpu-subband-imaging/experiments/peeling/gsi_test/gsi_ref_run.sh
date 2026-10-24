#!/usr/bin/env bash
# Reference comparison on identical prepared inputs: CPU TTCal (main pipeline commands),
# GPU production settings, GPU CPU-matched settings; same imaging; timings.
source /lustre/nkosogor/peel_exp/repo/experiments/peeling/env.sh
T=/lustre/nkosogor/peel_exp/gsi_test; R=$T/ref; G=$PX/gsi_branch
source /opt/miniconda3/etc/profile.d/conda.sh; conda activate $PY_ENV
export PYTHONPATH=$G/src GSI_ALLOWED_LUSTRE_WRITE_ROOT=$T
SNAPS=$(ls $R | grep MHz_)
for s in $SNAPS; do for v in cpu_sky cpu_sky_rfi gpu_prod gpu_cpumatch; do
  [ -d $R/$s/$v.ms ] || cp -r $R/$s/prep.ms $R/$s/$v.ms; done; done
# CPU TTCal: 6 snapshots at a time (single-threaded processes)
cpu_one() { local s=$1; local P=$PX/repo/experiments/peeling
  $P/cpu_zest.sh $R/$s/cpu_sky.ms sky 2>&1 | grep CPU_ZEST | sed "s/^/$s /"
  $P/cpu_zest.sh $R/$s/cpu_sky_rfi.ms sky 2>&1 | grep CPU_ZEST | sed "s/^/$s /"
  $P/cpu_zest.sh $R/$s/cpu_sky_rfi.ms rfi 2>&1 | grep CPU_ZEST | sed "s/^/$s /"; }
export -f cpu_one; export R PX
echo $SNAPS | tr ' ' '\n' | xargs -P 6 -I{} bash -c 'cpu_one {}' > $R/cpu_timing.log 2>&1
# GPU daemons with each config's ZEST_* settings (per band)
gpu() { # variant config ttcalx
  for b in 32 41 69; do
    env_args=$(python -c "
from gpu_subband_imaging.config import Config
c = Config.load('$T/$2').pipeline
e = {'ZEST_COLUMN': 'DATA', **c.peel_for_band($b).daemon_env(c.cal.rfi_sources)}
print(' '.join(f'{k}={v}' for k, v in e.items()))")
    env $env_args CUDA_VISIBLE_DEVICES=0 TTCALX_DIR=$3 JULIA_CUDA_USE_COMPAT=false \
        JULIA_DEPOT_PATH=$PX/depot:$SHARED_DEPOT $JULIA_BIN/julia --project=$3 $G/workers/zest_daemon.jl $SKY_MODEL \
        < <(ls -d $R/${b}MHz_*/$1.ms; echo QUIT) 2>&1 | grep -E "^ZESTED|^ZEST_TIMING|^FAILED|^DAEMON_READY|PEEL_SUMMARY" | sed "s/^/$1 $b /"
  done; }
gpu gpu_prod config-prod /opt/devel/nkosogor/nkosogor/TTCalX > $R/gpu_prod.log 2>&1
gpu gpu_cpumatch config-cpumatch $PX/ttcalx_branch > $R/gpu_cpumatch.log 2>&1
# Image everything with the GPU pipeline's WSClean settings
img_one() { local s=$1 v=$2; local band=${s%%MHz*}
  [ -f $R/$s/img/$v-I-image.fits ] || $PX/repo/experiments/peeling/image.sh $R/$s/$v.ms $R/$s/img/$v $band > /dev/null 2>&1; }
export -f img_one
for s in $SNAPS; do for v in prep cpu_sky cpu_sky_rfi gpu_prod gpu_cpumatch; do echo "$s $v"; done; done | \
  xargs -P 4 -n 2 bash -c 'WSC_THREADS=6 img_one $0 $1'
echo REF_DONE
