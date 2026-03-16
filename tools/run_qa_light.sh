#!/usr/bin/env bash
# =============================================================================
#  run_qa_light.sh — Run the full QA-Light pipeline
# =============================================================================
#  Orchestrates: fetch → flag/calibrate → image (lower band)
#
#  Usage:
#    ./run_qa_light.sh                    # Run everything (lower + upper fetch/cal)
#    ./run_qa_light.sh --lower-only       # Only process lower band subbands
#    ./run_qa_light.sh --fetch-only       # Only fetch data (no cal, no imaging)
#    ./run_qa_light.sh --skip-fetch       # Skip fetch, assume data is on NVMe
#    ./run_qa_light.sh --skip-imaging     # Fetch + calibrate, but no imaging
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

# ── Parse flags ──────────────────────────────────────────────────────────────
LOWER_ONLY=false
FETCH_ONLY=false
SKIP_FETCH=false
SKIP_IMAGING=false

for arg in "$@"; do
    case "$arg" in
        --lower-only)   LOWER_ONLY=true ;;
        --fetch-only)   FETCH_ONLY=true ;;
        --skip-fetch)   SKIP_FETCH=true ;;
        --skip-imaging) SKIP_IMAGING=true ;;
        --help|-h)
            echo "Usage: $0 [--lower-only] [--fetch-only] [--skip-fetch] [--skip-imaging]"
            echo ""
            echo "Options:"
            echo "  --lower-only    Only process lower-band subbands (18-41 MHz)"
            echo "  --fetch-only    Only fetch data to NVMe (no calibration/imaging)"
            echo "  --skip-fetch    Skip data fetch (assume already on NVMe)"
            echo "  --skip-imaging  Fetch + calibrate but skip wsclean imaging"
            exit 0
            ;;
        *) echo "Unknown option: $arg"; exit 1 ;;
    esac
done

echo "============================================================"
echo " QA-Light Pipeline"
echo " Date: ${OBS_DATE}  Hour: ${UTC_HOUR}  Timestep: ${TIMESTEP:-first}"
echo " Flag bad ants: ${FLAG_BAD_ANTS}  AOFlagger: ${RUN_AOFLAGGER}"
echo "============================================================"

# Decide which subbands to process
if ${LOWER_ONLY}; then
    PROCESS_SUBBANDS=("${LOWER_SUBBANDS[@]}")
else
    PROCESS_SUBBANDS=("${ALL_SUBBANDS[@]}")
fi

# ── Step 1: Fetch data ──────────────────────────────────────────────────────
if ! ${SKIP_FETCH}; then
    echo ""
    echo "========== Step 1: Fetch Data =========="
    # Temporarily override ALL_SUBBANDS if --lower-only
    if ${LOWER_ONLY}; then
        ALL_SUBBANDS=("${LOWER_SUBBANDS[@]}")
    fi
    bash "${SCRIPT_DIR}/01_fetch_data.sh"
fi

if ${FETCH_ONLY}; then
    echo ""
    echo "Done (--fetch-only)."
    exit 0
fi

# ── Step 2: Flag + Calibrate ────────────────────────────────────────────────
echo ""
echo "========== Step 2: Flag + Calibrate =========="

# Validate bandpass table
if [[ ! -d "${BP_TABLE}" ]]; then
    echo "[ERROR] Bandpass table not found: ${BP_TABLE}"
    echo "        Set BP_TABLE in config.sh to the full path of the .B.flagged table."
    exit 1
fi
echo "Bandpass table: ${BP_TABLE}"

CAL_FAILED=()
CAL_OK=()

for FREQ in "${PROCESS_SUBBANDS[@]}"; do
    MS_FILE=$(ls -d "${WORK_DIR}/${FREQ}"/*.ms 2>/dev/null | head -1 || true)
    if [[ -z "${MS_FILE}" ]]; then
        echo "[WARN] No MS for ${FREQ} — skipping calibration"
        continue
    fi

    echo ""
    echo "--- ${FREQ}: $(basename "${MS_FILE}") ---"

    # Build python args (same BP_TABLE for all subbands)
    CAL_ARGS=("${MS_FILE}" "${BP_TABLE}")

    if [[ "${FLAG_BAD_ANTS}" == "true" ]]; then
        CAL_ARGS+=("--flag-ants" "--mnc-conda-env" "${CONDA_ENV_MNC}")
    fi

    if [[ "${RUN_AOFLAGGER}" == "true" ]]; then
        CAL_ARGS+=("--aoflagger" "--aoflagger-strategy" "${AOFLAGGER_STRATEGY}" "--aoflagger-bin" "${AOFLAGGER_BIN}")
    fi

    if python "${SCRIPT_DIR}/02_apply_cal.py" "${CAL_ARGS[@]}"; then
        CAL_OK+=("${FREQ}")
    else
        CAL_FAILED+=("${FREQ}")
        echo "[FAIL] Calibration failed for ${FREQ}"
    fi
done

echo ""
echo "Calibration: OK=${CAL_OK[*]:-none}  FAILED=${CAL_FAILED[*]:-none}"

# ── Step 3: Image lower band ────────────────────────────────────────────────
if ! ${SKIP_IMAGING}; then
    echo ""
    echo "========== Step 3: Image Lower Band =========="
    bash "${SCRIPT_DIR}/03_image_lower_band.sh"
fi

# ── Step 4: Image upper band (skip if --lower-only) ─────────────────────────
if ! ${SKIP_IMAGING} && ! ${LOWER_ONLY}; then
    echo ""
    echo "========== Step 4: Image Upper Band =========="
    bash "${SCRIPT_DIR}/04_image_upper_band.sh"
fi

echo ""
echo "============================================================"
echo " QA-Light Pipeline Complete"
echo " Images:  ${OUTPUT_DIR}"
echo " NVMe:    ${WORK_DIR}"
echo "============================================================"

# ── Cleanup: remove NVMe working copy ───────────────────────────────────────
if ! ${FETCH_ONLY} && ! ${SKIP_IMAGING}; then
    echo ""
    echo "Cleaning up NVMe working directory: ${WORK_DIR}"
    rm -rf "${WORK_DIR}"
    echo "Done."
fi

