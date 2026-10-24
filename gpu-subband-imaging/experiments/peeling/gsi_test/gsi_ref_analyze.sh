#!/usr/bin/env bash
# Summaries of the reference set (identical inputs), an imaging-determinism check, timings.
source /lustre/nkosogor/peel_exp/repo/experiments/peeling/env.sh
R=/lustre/nkosogor/peel_exp/gsi_test/ref
source /opt/miniconda3/etc/profile.d/conda.sh; conda activate $PY_ENV
cd $PX_SCRIPTS
python study_summary.py $R --vis --out $R/summary.json > $R/summary.log 2>&1
python vpol_summary.py $R > $R/vpol.json 2>/dev/null
# Determinism: image one GPU output a second time and compare.
s=$(ls $R | grep 32MHz_ | head -1)
WSC_THREADS=6 ./image.sh $R/$s/gpu_cpumatch.ms /tmp/nk_det_again 32 > /dev/null 2>&1
python - $R/$s/img/gpu_cpumatch /tmp/nk_det_again <<'PY'
import sys, numpy as np
from astropy.io import fits
a = np.squeeze(fits.getdata(sys.argv[1] + "-I-image.fits")); b = np.squeeze(fits.getdata(sys.argv[2] + "-I-image.fits"))
print("DETERMINISM identical=%s max|diff|=%.3g rms=%.3g" % (np.array_equal(a, b), np.abs(a - b).max(), np.std(a)))
PY
rm -f /tmp/nk_det_again*
# MS-level: GPU CPU-matched daemon outputs are deterministic? compare DATA of two identical runs not available; report GPU vs CPU-mode later.
echo ANALYZE_DONE
