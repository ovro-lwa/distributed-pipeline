#!/usr/bin/env bash
# prep_one.sh SRC_MS OUT_DIR  (bandpass from $BP)
SRC=$1; OUT=$2
source "$(dirname "$0")/env.sh"
px_py "$PX_SCRIPTS/prep_ms.py" "$SRC" "$OUT" --bp "$BP" 2>&1 | tail -1
