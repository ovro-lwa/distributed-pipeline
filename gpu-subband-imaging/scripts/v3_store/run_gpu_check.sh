#!/bin/bash
# gsi v3 step 5 -- GPU check of the hour products from --archive vs from the MSes.
# Run on a node with a free GPU (not as the production user):
#     GPU=1 bash /lustre/nkosogor/peel_exp/v3/store/gpu_check/run_gpu_check.sh
# Needs ~5 GB of node-local disk under $W and ~15-30 min of one GPU.  Removes $W at the end
# unless KEEP=1.  Result: $W/out/<band>/result.json, summary line "ALL PASS" per band, and a
# copy of the result JSONs in $L/results_<host>_<date>/.
set -eo pipefail
GPU=${GPU:?set GPU=<device index>}
W=${W:-/tmp/nk_v3_gpucheck}
L=/lustre/nkosogor/peel_exp/v3/store/gpu_check
BANDS=${BANDS:-"23 55 82"}
VENV=/opt/devel/pipeline/gsi-v2/venv
PY38=/opt/devel/pipeline/envs/py38_orca_nkosogor/bin/python
NICE="nice -n 19 ionice -c3"
export CASARCFILES=/opt/devel/pipeline/gsi-v2/casarc XDG_CACHE_HOME=/lustre/nkosogor/peel_exp/e2e/xdg_cache
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 NUMPY_MADVISE_HUGEPAGE=0 CUPY_CACHE_DIR=/tmp/nk_cupy_cache

mkdir -p "$W" && cd "$W"
tar xf $L/pipeline_src.tar          # -> pipe/gpu-subband-imaging (exp/gsi-v3-store, incl. scripts/v3_store)
tar xf $L/imager_src.tar            # -> imager/python/ttcalx_imager (exp/imager-v3-store)
P=$W/pipe/gpu-subband-imaging/src; I=$W/imager/python/ttcalx_imager/src
cp $W/pipe/gpu-subband-imaging/scripts/v3_store/*.py .

# 0. the imager copy must shadow the venv's editable install
PYTHONPATH=$I $VENV/bin/python -c "import ttcalx_imager, sys; f = ttcalx_imager.__file__; print(f); sys.exit(0 if f.startswith('$I') else 1)"

# 1. calibration tables: read-only originals -> local copy, verified by file sizes and md5
BP=/lustre/pipeline/calibration/results/2026-10-31/21h/successful/20261031_030001/tables/calibration_2026-10-31_21h.B.flagged
XY=/lustre/gh/polcal/xyphase_delay_pos_3.8643ns.Xf
mkdir -p cal
for s in $BP $XY; do
  b=$(basename $s); [ -d cal/$b ] || cp -r $s cal/
  a=$(cd $s && find . -type f | sort | xargs md5sum | md5sum); c=$(cd cal/$b && find . -type f | sort | xargs md5sum | md5sum)
  [ "$a" == "$c" ] || { echo "cal copy mismatch $b"; exit 1; }
done

# 2. 12 averaged MSes per band with the v2 prep (stage_ms, fastcal.flag_calibrate + average,
#    fix_uvw centroid; no peel), bad antennas = the list fetched once on 2026-11-01 (badants.txt)
for b in $BANDS; do
  CUDA_VISIBLE_DEVICES= PYTHONPATH=$P $NICE $PY38 prep_ms.py $b 12 $L/badants.txt cal/$(basename $BP) cal/$(basename $XY) \
      $W/ms/${b}MHz $W/raw/$b > prep_$b.log 2>&1 &
done
wait

# 3. the archives, through the worker hook (ArchiveClient -> hourstore serve in the imager venv)
CUDA_VISIBLE_DEVICES= MS_ROOT=$W/ms PYTHONPATH=$P $NICE $PY38 equiv_build.py $W/gsiv $I $VENV/bin/python $BANDS | grep -v Meas
#    determinism: the records equal those built on calim10 on 2026-11-01 (gsiv_test)
PYTHONPATH=$P $VENV/bin/python - $W/gsiv /lustre/nkosogor/peel_exp/v3/store/gsiv_test <<'PY'
import sys, glob, os
import numpy as np
from gpu_subband_imaging.v3 import hourstore as hs
for f in sorted(glob.glob(sys.argv[1] + "/*.gsiv")):
    g = os.path.join(sys.argv[2], os.path.basename(f))
    if not os.path.exists(g):
        print("no reference for", f); continue
    _, _, ra = hs.read_records(f); _, _, rb = hs.read_records(g)
    strip = lambda m: {k: v for k, v in m.items() if k not in ("path", "written_ns")}   # MS path, write time differ
    same = len(ra) == len(rb) and all(
        strip(x.meta) == strip(y.meta) and x.arrays.keys() == y.arrays.keys()
        and all(x.arrays[k].tobytes() == y.arrays[k].tobytes() for k in x.arrays) for x, y in zip(ra, rb))
    print(os.path.basename(f), "records", len(ra), "identical to the 2026-11-01 build (except MS path, write time):", same)
PY

# 4. GPU: deterministic stages bitwise + products MSes x2 vs --archive x2
for b in $BANDS; do
  f=$(ls $W/gsiv/${b}MHz-*.gsiv)
  CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$I $VENV/bin/python gpu_equiv.py --band $b --ms-dir $W/ms/${b}MHz --archive $f \
      --out $W/out/$b --lst-hour 4 --products orca --reps 2 --ttcalx-image $VENV/bin/ttcalx-image 2>&1 | grep -v Meas | tee gpu_$b.log
done

R=$L/results_$(hostname -s)_$(date +%Y%m%d_%H%M)
mkdir -p $R && for b in $BANDS; do cp $W/out/$b/result.json $R/result_$b.json; cp gpu_$b.log $R/; done
echo "results: $R"
[ "${KEEP:-0}" == 1 ] || { cd /; rm -rf "$W"; }
