#!/usr/bin/env bash
# =============================================================================
#  QA-Light Pipeline — Configuration
# =============================================================================
#  Edit these variables before running.  All other scripts source this file.
# =============================================================================

# ---------- Observation ----------
OBS_DATE="2026-02-26"           # Date of the slow data (YYYY-MM-DD)
UTC_HOUR="03"                   # UTC hour directory to pick data from

# ---------- Timestep selection ----------
# Which 10-second MS file to pick.  "" = first available (alphabetical).
# Set to a glob/substring to pick a specific one, e.g. "20260226_030500"
TIMESTEP=""

# ---------- Subbands ----------
# Lower band: 18–41 MHz (excluding 13 MHz) — imaged by this pipeline
LOWER_SUBBANDS=("18MHz" "23MHz" "27MHz" "32MHz" "36MHz" "41MHz")

# Upper band: 46–82 MHz — fetched and calibrated but NOT imaged here
UPPER_SUBBANDS=("46MHz" "50MHz" "55MHz" "59MHz" "64MHz" "69MHz" "73MHz" "78MHz" "82MHz")

# All subbands = lower + upper (data is fetched/calibrated for all)
ALL_SUBBANDS=("${LOWER_SUBBANDS[@]}" "${UPPER_SUBBANDS[@]}")

# ---------- Data paths ----------
SLOW_DATA_ROOT="/lustre/pipeline/slow"
# Slow data layout:  ${SLOW_DATA_ROOT}/<FREQ>/<DATE>/<HH>/<MS files>

WORK_DIR="/fast/pipeline/qa_light/${OBS_DATE}_${UTC_HOUR}"
# Each subband gets:  ${WORK_DIR}/<FREQ>/<ms_file>

# ---------- Calibration ----------
# Bandpass calibration table — full path to the .B.flagged table.
# These live under /lustre/pipeline/calibration/results/ with structure:
#   /lustre/pipeline/calibration/results/<DATE>/<LST>h/successful/<RUN_TIMESTAMP>/tables/calibration_<DATE>_<LST>h.B.flagged
#
# Example:
#   /lustre/pipeline/calibration/results/2026-02-20/05h/successful/20260226_172229/tables/calibration_2026-02-20_05h.B.flagged
#
# You must specify the full path because the run timestamp directory varies.
BP_TABLE="/lustre/pipeline/calibration/results/2026-02-20/05h/successful/20260226_172229/tables/calibration_2026-02-20_05h.B.flagged"

# ---------- Flagging (all optional) ----------
# Set to "true" to enable, "false" to skip.
# Since slow data may already be pre-flagged (averaged pipeline) these are
# optional.  The main pipeline flags bad antennas per-MS (Phase 1) and runs
# AOFlagger on concatenated data (Phase 2).  Here we operate on single MS
# files, so both steps happen per-MS.
FLAG_BAD_ANTS="true"     # Flag bad antennas via mnc_python (needs 'development' conda env)
RUN_AOFLAGGER="true"     # Run AOFlagger RFI detection
AOFLAGGER_STRATEGY="/lustre/ghellbourg/AOFlagger_strat_opt/LWA_opt_GH1.lua"

# ---------- Imaging (lower band only) ----------
WSCLEAN_BIN="/opt/bin/wsclean"
AOFLAGGER_BIN="/opt/bin/aoflagger"

# MFS imaging parameters for one 10-s MS file per subband.
# Based on user spec (similar to the 10-subband combined example but per-subband).
# Stokes I only, no polarization.
WSCLEAN_NITER="10000"
WSCLEAN_MGAIN="0.85"
WSCLEAN_SIZE="4096"
WSCLEAN_SCALE="0.03125"       # degrees
WSCLEAN_WEIGHT="briggs 0"
WSCLEAN_HORIZON_MASK="10deg"
WSCLEAN_TAPER="3"             # taper-inner-tukey (matches cal pipeline)
WSCLEAN_MEM="50"
WSCLEAN_THREADS="10"          # -parallel-reordering & -j
WSCLEAN_AUTO_THRESHOLD="0.5"
WSCLEAN_AUTO_MASK="3"

# ---------- Output ----------
OUTPUT_DIR="/lustre/pipeline/qa_light_images/${OBS_DATE}/${UTC_HOUR}"

# ---------- Conda environments ----------
CONDA_ENV_PIPELINE="py38_orca_nkosogor"   # Main orca/CASA environment
CONDA_ENV_MNC="development"               # For mnc_python bad-antenna lookup
