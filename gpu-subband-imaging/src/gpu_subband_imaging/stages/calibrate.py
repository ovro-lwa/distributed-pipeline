"""Apply bandpass and XY-phase tables using frequency-matched SPWs."""
from __future__ import annotations

from pathlib import Path

# CASA runs in a subprocess because its Python tools are not thread-safe.
_SNIPPET = """
import numpy as np
from casacore.tables import table
from casatasks import applycal, flagdata
from gpu_subband_imaging.subbands import spwmap

ms, bp, xy = {ms!r}, {bp!r}, {xy!r}
with table(ms + "/SPECTRAL_WINDOW", ack=False) as t:
    if np.max(t.getcol("CHAN_FREQ")) > 85e6:
        flagdata(vis=ms, mode="manual", spw="*:85.0~100.0MHz", flagbackup=False)
bp_map = spwmap(ms, bp)
xy_map = spwmap(ms, xy)
applycal(vis=ms, gaintable=[bp, xy], spwmap=[bp_map, xy_map],
         flagbackup=False, calwt=False)
print("SPWMAP", bp_map, xy_map)
"""


def applycal(ms: Path, bp_table: str, xy_table: str) -> None:
    from .casarun import run_py
    run_py(_SNIPPET.format(ms=str(ms), bp=bp_table, xy=xy_table))


# With AOFlagger off, combine CASA work to avoid three interpreter startups.
_ALL_SNIPPET = """
import numpy as np
from casacore.tables import table
from casatasks import applycal, flagdata, mstransform
from gpu_subband_imaging.subbands import spwmap

ms, bp, xy, out, ba, chanbin = {ms!r}, {bp!r}, {xy!r}, {out!r}, {ba!r}, {chanbin}
if ba:
    flagdata(vis=ms, mode="manual", antenna=ba, flagbackup=False)
with table(ms + "/SPECTRAL_WINDOW", ack=False) as t:
    if np.max(t.getcol("CHAN_FREQ")) > 85e6:
        flagdata(vis=ms, mode="manual", spw="*:85.0~100.0MHz", flagbackup=False)
bp_map = spwmap(ms, bp)
xy_map = spwmap(ms, xy)
applycal(vis=ms, gaintable=[bp, xy], spwmap=[bp_map, xy_map],
         flagbackup=False, calwt=False)
mstransform(vis=ms, outputvis=out, datacolumn="corrected",
            chanaverage=True, chanbin=chanbin)
print("SPWMAP", bp_map, xy_map)
"""


def flag_cal_average(ms: Path, out_ms: Path, bp_table: str, xy_table: str,
                     badants_csv: str, chanbin: int) -> Path:
    from .casarun import run_py
    run_py(_ALL_SNIPPET.format(ms=str(ms), bp=bp_table, xy=xy_table,
                               out=str(out_ms), ba=badants_csv, chanbin=chanbin))
    return out_ms
