#!/usr/bin/env python3
"""02_apply_cal.py — Flag bad antennas (optional) + apply bandpass calibration.

This mirrors the main pipeline's Phase 1 logic:
  1. (Optional) Flag bad antennas via mnc_python
  2. Apply bandpass calibration via CASA applycal

Usage:
    python 02_apply_cal.py <ms_path> <bp_table> [--flag-ants] [--aoflagger] [--aoflagger-strategy PATH]

The script is designed to be called from the shell wrapper for each MS file.
It can also be run standalone.
"""
import argparse
import json
import logging
import os
import re
import subprocess
import sys

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ── Bandpass-only calibration (no XY-phase) ──────────────────────────────────

def calculate_spwmap(ms_path: str, caltable_path: str):
    """Compute SPW mapping between an MS and a calibration table.

    Mirrors orca.transform.subband_processing.calculate_spwmap.
    """
    try:
        import casacore.tables as pt
    except ImportError:
        # Fall back: try python-casacore installed elsewhere
        raise ImportError("casacore.tables (python-casacore) is required")

    try:
        with pt.table(os.path.join(ms_path, "SPECTRAL_WINDOW"), ack=False) as t:
            ms_freqs = [t.getcell("CHAN_FREQ", i) for i in range(t.nrows())]
        with pt.table(os.path.join(caltable_path, "SPECTRAL_WINDOW"), ack=False) as t:
            cal_freqs = [t.getcell("CHAN_FREQ", i) for i in range(t.nrows())]

        spwmap = []
        for ms_f in ms_freqs:
            ms_min, ms_max = np.min(ms_f), np.max(ms_f)
            best_match, best_overlap = -1, 0.0
            for cal_idx, cal_f in enumerate(cal_freqs):
                cal_min, cal_max = np.min(cal_f), np.max(cal_f)
                overlap = min(ms_max, cal_max) - max(ms_min, cal_min)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = cal_idx
            if best_match == -1:
                ms_center = np.mean(ms_f)
                diffs = [abs(np.mean(cf) - ms_center) for cf in cal_freqs]
                best_match = int(np.argmin(diffs))
            spwmap.append(best_match)
        return spwmap
    except Exception as e:
        logger.error(f"Error calculating SPW map: {e}")
        return None


def apply_bandpass(ms_path: str, bp_table: str) -> bool:
    """Apply bandpass-only calibration to an MS (no XY-phase).

    Mirrors orca.transform.subband_processing.apply_calibration but with
    only the bandpass table.
    """
    bp_map = calculate_spwmap(ms_path, bp_table)
    if bp_map is None:
        logger.error("Failed to map SPWs — cannot apply calibration")
        return False

    python_code = f"""
import sys
from casatasks import clearcal, applycal
try:
    clearcal(vis='{ms_path}', addmodel=False)
    applycal(vis='{ms_path}', gaintable=['{bp_table}'],
             spwmap=[{bp_map}], flagbackup=False, calwt=False)
except Exception as e:
    print(f"CASA Error: {{e}}")
    sys.exit(1)
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", python_code],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            logger.error(f"Calibration subprocess failed:\n{result.stderr}")
            return False
        logger.info(f"Bandpass applied successfully to {os.path.basename(ms_path)}")
        return True
    except Exception as e:
        logger.error(f"Subprocess launch failed: {e}")
        return False


# ── Bad antenna flagging (optional, mirrors main pipeline) ───────────────────

def get_bad_antenna_numbers(ms_path: str, conda_env: str = "development"):
    """Query mnc_python for bad antennas at the observation time.

    Mirrors orca.transform.subband_processing.get_bad_antenna_numbers.
    Returns a list of bad correlator numbers, or empty list on failure.
    """
    try:
        import casacore.tables as pt
        with pt.table(os.path.join(ms_path, "OBSERVATION"), ack=False) as t:
            time_range = t.getcol("TIME_RANGE")[0]
            obs_mjd = float(time_range[0]) / 86400.0
    except Exception as e:
        logger.warning(f"Could not read observation time: {e}")
        return []

    # Use the helper script from orca if available, otherwise inline
    orca_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "orca")
    )
    helper_script = os.path.join(orca_root, "utils", "mnc_antennas.py")

    if os.path.exists(helper_script):
        cmd = ["conda", "run", "-n", conda_env, "python", helper_script, str(obs_mjd)]
    else:
        # Inline fallback
        code = f"""
import json
try:
    from mnc import anthealth
    from astropy.time import Time
    import lwa_antpos.mapping as mapping
    b = anthealth.get_badants('selfcorr', time={obs_mjd})
    badnames = b[1]
    correlators = [
        mapping.antname_to_correlator(name.rstrip('A').rstrip('B'))
        for name in badnames
    ]
    print(json.dumps({{"bad_correlator_numbers": sorted(set(correlators))}}))
except Exception as e:
    print(json.dumps({{"bad_correlator_numbers": [], "error": str(e)}}))
"""
        cmd = ["conda", "run", "-n", conda_env, "python", "-c", code]

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        for line in reversed(res.stdout.strip().split("\n")):
            try:
                data = json.loads(line)
                if data is not None:
                    bad_ants = data.get("bad_correlator_numbers", [])
                    logger.info(
                        "MNC antenna health: requested MJD=%.6f, "
                        "matched MJD=%s, bad=%d",
                        obs_mjd,
                        data.get("data_timestamp_mjd"),
                        len(bad_ants),
                    )
                    return bad_ants
            except (json.JSONDecodeError, TypeError):
                continue
    except Exception as e:
        logger.warning(f"Bad-antenna lookup failed (non-fatal): {e}")
    return []


def flag_bad_antennas(ms_path: str, conda_env: str = "development") -> bool:
    """Flag bad antennas in an MS. Returns True if any were flagged."""
    bad_ants = get_bad_antenna_numbers(ms_path, conda_env)
    if not bad_ants:
        logger.info("No bad antennas to flag (or lookup unavailable)")
        return False

    import casacore.tables as pt
    with pt.table(os.path.join(ms_path, "ANTENNA"), ack=False) as t:
        n_antennas = t.nrows()
    if n_antennas and len(bad_ants) >= 0.5 * n_antennas:
        raise RuntimeError(
            f"Refusing to flag {len(bad_ants)}/{n_antennas} antennas "
            f"({len(bad_ants) / n_antennas:.1%}); MNC lookup is likely invalid"
        )

    bad_ant_str = ",".join(map(str, bad_ants))
    logger.info(f"Flagging bad antennas: {bad_ant_str}")
    python_code = f"""
from casatasks import flagdata
flagdata(vis='{ms_path}', mode='manual', antenna='{bad_ant_str}', flagbackup=False)
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", python_code],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            logger.warning(f"Antenna flagging failed (non-fatal): {result.stderr}")
            return False
        return True
    except Exception as e:
        logger.warning(f"Antenna flagging failed (non-fatal): {e}")
        return False


# ── AOFlagger (optional) ────────────────────────────────────────────────────

def run_aoflagger(ms_path: str, strategy: str, aoflagger_bin: str = "/opt/bin/aoflagger") -> bool:
    """Run AOFlagger on a single MS. Returns True on success."""
    if not os.path.exists(aoflagger_bin):
        logger.warning(f"AOFlagger binary not found at {aoflagger_bin} — skipping")
        return False
    if not os.path.exists(strategy):
        logger.warning(f"AOFlagger strategy not found at {strategy} — skipping")
        return False

    cmd = [aoflagger_bin, "-strategy", strategy, ms_path]
    logger.info(f"Running AOFlagger: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning(f"AOFlagger returned non-zero (non-fatal): {result.stderr[:500]}")
            return False
        logger.info(f"AOFlagger completed on {os.path.basename(ms_path)}")
        return True
    except Exception as e:
        logger.warning(f"AOFlagger failed (non-fatal): {e}")
        return False


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Apply bandpass calibration (optionally flag first) to a single MS",
    )
    parser.add_argument("ms_path", help="Path to the measurement set")
    parser.add_argument("bp_table", help="Path to the bandpass calibration table")
    parser.add_argument("--flag-ants", action="store_true",
                        help="Flag bad antennas via mnc_python before calibration")
    parser.add_argument("--mnc-conda-env", default="development",
                        help="Conda env for mnc_python (default: development)")
    parser.add_argument("--aoflagger", action="store_true",
                        help="Run AOFlagger after calibration")
    parser.add_argument(
        "--aoflagger-strategy",
        default="/lustre/ghellbourg/AOFlagger_strat_opt/LWA_opt_GH1.lua",
        help="AOFlagger strategy file",
    )
    parser.add_argument("--aoflagger-bin", default="/opt/bin/aoflagger",
                        help="Path to aoflagger binary")
    args = parser.parse_args()

    ms = os.path.abspath(args.ms_path)
    bp = os.path.abspath(args.bp_table)

    if not os.path.isdir(ms):
        logger.error(f"MS not found: {ms}")
        sys.exit(1)
    if not os.path.isdir(bp):
        logger.error(f"Bandpass table not found: {bp}")
        sys.exit(1)

    # Step 1 (optional): Flag bad antennas
    if args.flag_ants:
        logger.info("=== Flagging bad antennas ===")
        flag_bad_antennas(ms, args.mnc_conda_env)

    # Step 2: Apply bandpass calibration
    logger.info("=== Applying bandpass calibration ===")
    ok = apply_bandpass(ms, bp)
    if not ok:
        logger.error("Bandpass calibration FAILED")
        sys.exit(1)

    # Step 3 (optional): AOFlagger
    if args.aoflagger:
        logger.info("=== Running AOFlagger ===")
        run_aoflagger(ms, args.aoflagger_strategy, args.aoflagger_bin)

    logger.info("Done.")


if __name__ == "__main__":
    main()
