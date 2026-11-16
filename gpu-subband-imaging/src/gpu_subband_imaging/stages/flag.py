"""Flag bad antennas and radio-frequency interference."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple

_BADANT_PREFIX = "GSI_BADANTS="
_BADANT_SNIPPET = """
from mnc import anthealth
from lwa_antpos import mapping
c, bad = anthealth.get_badants('selfcorr', time={mjd})
print('{prefix}' + ','.join(str(x) for x in sorted(set(
    mapping.antname_to_correlator(b.rstrip('AB')) for b in bad))))
"""


def badants(mjd: float, dev_python: str, cache: Optional[Dict[float, str]] = None
            ) -> str:
    """Return bad correlator numbers as CSV, or an empty string on failure."""
    if cache is not None and mjd in cache:
        return cache[mjd]
    val = ""
    for _ in range(2):
        code = _BADANT_SNIPPET.format(mjd=mjd, prefix=_BADANT_PREFIX)
        p = subprocess.run([dev_python, "-c", code],
                           capture_output=True, text=True)
        result = next(
            (line[len(_BADANT_PREFIX):].strip()
             for line in p.stdout.splitlines()
             if line.startswith(_BADANT_PREFIX)),
            None,
        )
        if p.returncode == 0 and result is not None:
            val = result
            break
    if cache is not None:
        cache[mjd] = val
    return val


_BADANT_REC_SNIPPET = """
from mnc import anthealth
from lwa_antpos import mapping
c, bad = anthealth.get_badants('selfcorr', time={mjd})
print('{prefix}' + str(c) + ' ' + ','.join(str(x) for x in sorted(set(
    mapping.antname_to_correlator(b.rstrip('AB')) for b in bad))))
"""


def badants_record(mjd: float, dev_python: str) -> Tuple[Optional[float], str]:
    """(MJD key of the anthealth record used, bad correlator numbers as CSV) for `mjd`, or (None, "")
    on failure (an existing record with no bad antennas gives (key, "")).  The lookup returns the newest daily record keyed at or before `mjd` (records are keyed
    at ~08:00 UTC of their date and written hours later), so the answer for a night changes once that
    night's record is written: the nightly pins one list per night (scripts/run_nightly_lst.py)."""
    code = _BADANT_REC_SNIPPET.format(mjd=mjd, prefix=_BADANT_PREFIX)
    for _ in range(2):
        try:
            p = subprocess.run([dev_python, "-c", code], capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.TimeoutExpired):     # missing Python, hung etcd: a failed lookup
            continue
        line = next((ln[len(_BADANT_PREFIX):].strip() for ln in p.stdout.splitlines()
                     if ln.startswith(_BADANT_PREFIX)), None)
        if p.returncode == 0 and line:
            key, _, csv = line.partition(" ")
            try:
                return (float(key) if key not in ("None", "") else None), csv.strip()
            except ValueError:
                pass
    return None, ""


def flag_badants(ms: Path, badants_csv: str) -> None:
    if not badants_csv:
        return
    from .casarun import run_py
    run_py(f"from casatasks import flagdata\n"
           f"flagdata(vis='{ms}', mode='manual', antenna='{badants_csv}', "
           f"flagbackup=False)")


def aoflag(ms: Path, strategy: str, threads: int, aoflagger_bin: str = "aoflagger"
           ) -> None:
    subprocess.run([aoflagger_bin, "-j", str(threads), "-strategy", strategy,
                    str(ms)], check=True, capture_output=True)
