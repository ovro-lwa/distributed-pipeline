#!/usr/bin/env bash
# =============================================================================
#  01_fetch_data.sh — Pick one 10-s timestep per subband and copy to /fast
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

echo "=============================================="
echo " QA-Light: Fetch Data"
echo " Date=${OBS_DATE}  Hour=${UTC_HOUR}"
echo " Work dir: ${WORK_DIR}"
echo "=============================================="

mkdir -p "${WORK_DIR}"

for FREQ in "${ALL_SUBBANDS[@]}"; do
    SRC_DIR="${SLOW_DATA_ROOT}/${FREQ}/${OBS_DATE}/${UTC_HOUR}"

    if [[ ! -d "${SRC_DIR}" ]]; then
        echo "[WARN] Source directory missing: ${SRC_DIR} — skipping ${FREQ}"
        continue
    fi

    # Pick one MS file
    if [[ -n "${TIMESTEP}" ]]; then
        # User specified a timestep substring/glob
        MS_FILE=$(ls -d "${SRC_DIR}"/*"${TIMESTEP}"*"${FREQ}"*.ms 2>/dev/null | head -1 || true)
    else
        # First available (alphabetical = earliest)
        MS_FILE=$(ls -d "${SRC_DIR}"/*"${FREQ}"*.ms 2>/dev/null | head -1 || true)
    fi

    if [[ -z "${MS_FILE}" ]]; then
        echo "[WARN] No MS file found for ${FREQ} in ${SRC_DIR} — skipping"
        continue
    fi

    DEST_DIR="${WORK_DIR}/${FREQ}"
    mkdir -p "${DEST_DIR}"
    DEST="${DEST_DIR}/$(basename "${MS_FILE}")"

    if [[ -d "${DEST}" ]]; then
        echo "[SKIP] ${FREQ}: $(basename "${MS_FILE}") already on NVMe"
    else
        echo "[COPY] ${FREQ}: $(basename "${MS_FILE}") → ${DEST_DIR}/"
        cp -r "${MS_FILE}" "${DEST}"
    fi
done

echo ""
echo "Data fetch complete. Contents of ${WORK_DIR}:"
for FREQ in "${ALL_SUBBANDS[@]}"; do
    if [[ -d "${WORK_DIR}/${FREQ}" ]]; then
        N=$(ls -d "${WORK_DIR}/${FREQ}"/*.ms 2>/dev/null | wc -l || echo 0)
        echo "  ${FREQ}: ${N} MS file(s)"
    fi
done
