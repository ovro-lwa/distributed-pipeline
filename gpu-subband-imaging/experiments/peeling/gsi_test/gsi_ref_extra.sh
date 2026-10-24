#!/usr/bin/env bash
# 69 MHz with the RFI pass (as the main pipeline runs it on every band); Stokes V for all variants.
source /lustre/nkosogor/peel_exp/repo/experiments/peeling/env.sh
T=/lustre/nkosogor/peel_exp/gsi_test; R=$T/ref; G=$PX/gsi_branch
source /opt/miniconda3/etc/profile.d/conda.sh; conda activate $PY_ENV
export PYTHONPATH=$G/src GSI_ALLOWED_LUSTRE_WRITE_ROOT=$T
for s in $(ls $R | grep 69MHz_); do [ -d $R/$s/gpu_cpumatch_rfi.ms ] || cp -r $R/$s/prep.ms $R/$s/gpu_cpumatch_rfi.ms; done
env_args=$(python -c "
from gpu_subband_imaging.config import Config
c = Config.load('$T/config-cpumatch').pipeline
e = {'ZEST_COLUMN': 'DATA', **c.peel_for_band(32).daemon_env(c.cal.rfi_sources)}
print(' '.join(f'{k}={v}' for k, v in e.items()))")
env $env_args CUDA_VISIBLE_DEVICES=0 TTCALX_DIR=$PX/ttcalx_branch JULIA_CUDA_USE_COMPAT=false \
    JULIA_DEPOT_PATH=$PX/depot:$SHARED_DEPOT $JULIA_BIN/julia --project=$PX/ttcalx_branch $G/workers/zest_daemon.jl $SKY_MODEL \
    < <(ls -d $R/69MHz_*/gpu_cpumatch_rfi.ms; echo QUIT) 2>&1 | grep -E "^ZESTED|^FAILED|^ZEST_TIMING" > $R/gpu_cpumatch_rfi.log
for s in $(ls $R | grep 69MHz_); do
  [ -f $R/$s/img/gpu_cpumatch_rfi-I-image.fits ] || WSC_THREADS=12 $PX_SCRIPTS/image.sh $R/$s/gpu_cpumatch_rfi.ms $R/$s/img/gpu_cpumatch_rfi 69 > /dev/null 2>&1
done
cd $PX_SCRIPTS
python study_summary.py $R --vis --only gpu_cpumatch_rfi --snaps $(ls $R | grep 69MHz_) --out $R/summary_69rfi.json > /dev/null 2>&1
sed -i 's/VARS = \["prep", "cpu_sky", "cpu_sky_rfi", "gpu_prod", "gpu_cm_exact", "gpu_cm_exact_joint",/VARS = ["prep", "cpu_sky", "cpu_sky_rfi", "gpu_prod", "gpu_cpumatch", "gpu_cpumatch_rfi", "gpu_cm_exact", "gpu_cm_exact_joint",/' vpol_summary.py
python vpol_summary.py $R > $R/vpol.json 2>/dev/null
echo EXTRA_DONE
