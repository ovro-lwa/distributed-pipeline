#!/usr/bin/env python3
"""Helper script to query bad antennas from the MNC antenna-health database.

This script is executed as a **subprocess** in the ``development`` conda
environment (which has ``mnc`` and ``lwa_antpos`` installed).  The main
pipeline calls it via::

    conda run -n development python -m orca.utils.mnc_antennas <MJD>

Communication is via JSON on stdout; errors go to stderr.
"""
import argparse
import json
import sys
import traceback
from astropy.time import Time

# Attempt imports specific to the 'development' environment
_mnc_available = False
_import_error = None
try:
    from mnc import anthealth
    import lwa_antpos.mapping as mapping
    _mnc_available = True
except ImportError as e:
    _import_error = str(e)


def get_bad_correlators_from_anthealth(mjd_time: float):
    """Query anthealth.get_badants and map signal names to correlator numbers.

    Args:
        mjd_time: Observation time as Modified Julian Date.

    Returns:
        Tuple of (sorted list of bad correlator numbers, closest MJD float).
    """
    # Suppress Astropy ERFA warnings (like "dubious year")
    try:
        import warnings
        from astropy.utils.exceptions import ErfaWarning
        warnings.filterwarnings('ignore', category=ErfaWarning)
    except ImportError:
        pass

    try:
        time_obj = Time(mjd_time, format='mjd')
    except Exception as e:
        print(f"ERROR: Failed to parse MJD time {mjd_time}: {e}", file=sys.stderr)
        raise

    try:
        closest_mjd, bad_signal_names = anthealth.get_badants('selfcorr', time=time_obj.mjd)
    except Exception as e:
        print(f"ERROR: anthealth.get_badants failed with MJD {time_obj.mjd}: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        raise

    correlators = []
    for name in bad_signal_names:
        try:
            ant_name = name.rstrip('A').rstrip('B')
            corr_num = mapping.antname_to_correlator(ant_name)
            correlators.append(int(corr_num))
        except Exception as e:
            print(f"WARNING: Could not map antenna name '{name}' to correlator: {e}",
                  file=sys.stderr)

    return sorted(list(set(correlators))), float(closest_mjd)


def main():
    parser = argparse.ArgumentParser(
        description="Query bad antennas from MNC antenna-health database.",
    )
    parser.add_argument("mjd_time", type=float, help="Observation MJD time.")
    args = parser.parse_args()

    if not _mnc_available:
        print(
            f"ERROR_ENV: MNC/ORCA modules not available.  "
            f"Ensure 'mnc' and 'lwa_antpos' are installed.  "
            f"Import Error: {_import_error}",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        bad_correlator_numbers, data_timestamp_mjd = get_bad_correlators_from_anthealth(
            args.mjd_time,
        )
        output = {
            "requested_mjd": args.mjd_time,
            "data_timestamp_mjd": data_timestamp_mjd,
            "bad_correlator_numbers": bad_correlator_numbers,
        }
        print(json.dumps(output))
    except Exception:
        sys.exit(1)


if __name__ == "__main__":
    main()
