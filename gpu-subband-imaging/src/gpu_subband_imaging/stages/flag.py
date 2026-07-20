"""Flag bad antennas and radio-frequency interference."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict, Optional

_BADANT_SNIPPET = """
from mnc import anthealth
from lwa_antpos import mapping
c, bad = anthealth.get_badants('selfcorr', time={mjd})
print(','.join(str(x) for x in sorted(set(
    mapping.antname_to_correlator(b.rstrip('AB')) for b in bad))))
"""


def badants(mjd: float, dev_python: str, cache: Optional[Dict[float, str]] = None
            ) -> str:
    """Return bad correlator numbers as CSV, or an empty string on failure."""
    if cache is not None and mjd in cache:
        return cache[mjd]
    for _ in range(2):
        p = subprocess.run([dev_python, "-c", _BADANT_SNIPPET.format(mjd=mjd)],
                           capture_output=True, text=True)
        val = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
        if val:
            break
    if cache is not None:
        cache[mjd] = val
    return val


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
