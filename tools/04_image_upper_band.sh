#!/usr/bin/env bash
# =============================================================================
#  04_image_upper_band.sh — Combined wideband MFS imaging (>41 MHz)
# =============================================================================
#  Collects one calibrated MS per upper-band subband (same timestep) and
#  passes them all to a single wsclean call for wideband MFS synthesis.
#  Stokes I only.
#
#  The data should already be fetched + calibrated by run_qa_light.sh.
#
#  Usage:
#    # Image all upper-band subbands combined
#    ./04_image_upper_band.sh
#
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

echo "=============================================="
echo " QA-Light: Combined MFS Imaging (Upper Band)"
echo " Subbands: ${UPPER_SUBBANDS[*]}"
echo " Output:   ${OUTPUT_DIR}"
echo "=============================================="

mkdir -p "${OUTPUT_DIR}"

# ── Collect one MS file per subband ──────────────────────────────────────────
MS_FILES=()
MISSING=()

for FREQ in "${UPPER_SUBBANDS[@]}"; do
    MS_FILE=$(ls -d "${WORK_DIR}/${FREQ}"/*.ms 2>/dev/null | head -1 || true)
    if [[ -z "${MS_FILE}" ]]; then
        echo "[WARN] No MS found for ${FREQ} — will be excluded"
        MISSING+=("${FREQ}")
    else
        MS_FILES+=("${MS_FILE}")
        echo "  ${FREQ}: $(basename "${MS_FILE}")"
    fi
done

N_BANDS=${#MS_FILES[@]}
if [[ ${N_BANDS} -eq 0 ]]; then
    echo "[ERROR] No MS files found for any upper-band subband — aborting"
    exit 1
fi

echo ""
echo "Imaging ${N_BANDS} subbands combined (channels-out=${N_BANDS})"

IMG_NAME="${OUTPUT_DIR}/upper-band-QA-MFS"

# ── Build wsclean command ────────────────────────────────────────────────────
CMD=(
    "${WSCLEAN_BIN}"
    -j "${WSCLEAN_THREADS}"
    -pol I
    -size "${WSCLEAN_SIZE}" "${WSCLEAN_SIZE}"
    -scale "${WSCLEAN_SCALE}"
    -niter "${WSCLEAN_NITER}"
    -mgain "${WSCLEAN_MGAIN}"
    -parallel-reordering "${WSCLEAN_THREADS}"
    -weight ${WSCLEAN_WEIGHT}
    -horizon-mask "${WSCLEAN_HORIZON_MASK}"
    -taper-inner-tukey "${WSCLEAN_TAPER}"
    -mem "${WSCLEAN_MEM}"
    -channels-out "${N_BANDS}"
    -join-channels
    -fit-spectral-pol 4
    -local-rms
    -auto-threshold "${WSCLEAN_AUTO_THRESHOLD}"
    -auto-mask "${WSCLEAN_AUTO_MASK}"
    -no-update-model-required
    -name "${IMG_NAME}"
    "${MS_FILES[@]}"
)

export OPENBLAS_NUM_THREADS=1

echo ""
echo "[CMD] ${CMD[*]}"
echo ""

if "${CMD[@]}"; then
    echo ""
    echo "=============================================="
    echo " Upper-band combined MFS imaging complete"
    echo " Output: ${IMG_NAME}-MFS-image.fits (+ per-channel)"
    echo " Subbands used: ${N_BANDS}"
    if [[ ${#MISSING[@]} -gt 0 ]]; then
        echo " Missing (excluded): ${MISSING[*]}"
    fi
    echo "=============================================="
else
    echo ""
    echo "[FAIL] Combined MFS imaging failed (exit code $?)"
    exit 1
fi
