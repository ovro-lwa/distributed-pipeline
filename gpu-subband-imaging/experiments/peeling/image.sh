#!/usr/bin/env bash
# Dirty Stokes I/V snapshot with the GPU pipeline's WSClean settings
# (stages/image.py + tiers in config-allsubbands-auto).
#   image.sh MS OUT_PREFIX BAND_MHZ [extra wsclean args]
set -eo pipefail
MS=$1; OUT=$2; BAND=$3; shift 3
source "$(dirname "$0")/env.sh"
if   [ "$BAND" -le 36 ]; then NPIX=1507; SCALE=0.0767
elif [ "$BAND" -le 59 ]; then NPIX=2357; SCALE=0.049
else                          NPIX=3122; SCALE=0.037
fi
mkdir -p "$(dirname "$OUT")"
"$WSCLEAN" -j ${WSC_THREADS:-8} -pol IV -niter 0 -mem 20 -no-dirty \
    -no-update-model-required -size $NPIX $NPIX -scale $SCALE \
    -weight briggs 0 -taper-inner-tukey 30 -name "$OUT" "$@" "$MS" > "$OUT.wsclean.log" 2>&1
ls "$OUT"-I-image.fits "$OUT"-V-image.fits
