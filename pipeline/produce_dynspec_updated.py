"""Updated dynamic spectrum production pipeline.

This script produces dynamic spectra with improved timestamp parsing
and directory iteration. Processes averaged nighttime data across
all frequencies.
"""
from pathlib import Path
from datetime import datetime, timedelta
from celery import group

from orca.transform.spectrum_v2 import dynspec_reduce_v2, dynspec_map_v2

ROOT = Path("/lustre/pipeline/night-time/averaged")
FREQS = [
    "13MHz","18MHz","23MHz","27MHz","32MHz","36MHz",
    "41MHz","46MHz","50MHz","55MHz","59MHz","64MHz",
    "69MHz","73MHz","78MHz","82MHz",
]
OUT_DIR = "/lustre/pipeline/dynspec"
USE_MS_FLAG = False

def ts_from_ms(ms_path: Path) -> str:
    # 20251224_040009_13MHz_averaged.ms -> 20251224_040009
    parts = ms_path.stem.split("_", 2)
    return parts[0] + "_" + parts[1]

def dt_from_ts(ts: str) -> datetime:
    # naive datetime; treat as UTC consistently (same as your current code)
    return datetime.strptime(ts, "%Y%m%d_%H%M%S")

def iter_ms_paths(date: str, spw: str):
    """List MS under ROOT/spw/date/<HH>/*.ms (no deep recursion)."""
    base = ROOT / spw / date
    if not base.is_dir():
        return []
    return sorted(base.glob("*/*.ms"))

def main(DATE: str) -> None:
    t0 = datetime.now()
    print(f"[{t0:%F %T}] DATE={DATE}: Building global union timeline …")

    # 1) Union timeline across ALL SPWs
    all_ts: set[str] = set()
    ms_by_spw = {}  # keep for reuse so we don't rescan again
    for spw in FREQS:
        ms_list = iter_ms_paths(DATE, spw)
        ms_by_spw[spw] = ms_list
        for ms in ms_list:
            all_ts.add(ts_from_ms(ms))

    ts_list = sorted(all_ts)
    if not ts_list:
        print(f"[{datetime.now():%F %T}] No MS found for DATE={DATE}. Skipping.")
        return

    ts_index = {ts: i for i, ts in enumerate(ts_list)}

    # ✅ IMPORTANT FIX: start_ts from the earliest scan in the union
    start_ts = dt_from_ts(ts_list[0])

    print(f"Timeline contains {len(ts_list)} distinct scans (UNION across SPWs)")
    print(f"Timeline start={ts_list[0]} (start_ts={start_ts.isoformat()}Z)")
    print(f"Timeline end  ={ts_list[-1]}")

    # 2) Build map tasks (only for MS that exist; missing scans for an SPW naturally mean gaps)
    map_sigs = []
    for sb, spw in enumerate(FREQS):
        for ms in ms_by_spw[spw]:
            ts = ts_from_ms(ms)
            scan_no = ts_index[ts]
            map_sigs.append(
                dynspec_map_v2.si(
                    subband_no=sb,
                    scan_no=scan_no,
                    ms=str(ms),
                    bcal=None,
                    use_ms_flags=USE_MS_FLAG,
                )
            )

    print(f"Created {len(map_sigs)} map signatures")

    # 3) Submit maps
    map_result = group(map_sigs).apply_async()
    print(f"[{datetime.now():%F %T}] Submitted map group: {map_result.id}")

    # 4) Wait for maps
    spectra_lists_raw = map_result.get(disable_sync_subtasks=False, propagate=False)
    num_total = len(spectra_lists_raw)
    num_failed = sum(isinstance(r, Exception) for r in spectra_lists_raw)
    num_success = num_total - num_failed
    spectra_lists = [r for r in spectra_lists_raw if not isinstance(r, Exception)]
    print(f"[{datetime.now():%F %T}] Maps complete — {num_success}/{num_total} succeeded, {num_failed} failed")

    # 5) Reduce
    reduce_result = dynspec_reduce_v2.apply_async(args=(spectra_lists, start_ts, OUT_DIR))
    print(f"[{datetime.now():%F %T}] Submitted reduce task: {reduce_result.id}")
    reduce_result.get()

    t1 = datetime.now()
    print(f"[{t1:%F %T}] All done! FITS cubes in: {OUT_DIR}")
    print(f"Total elapsed: {t1 - t0}")


import time
from datetime import datetime, timedelta

if __name__ == "__main__":
    start_date = datetime.strptime("2025-12-25", "%Y-%m-%d").date()
    end_date   = datetime.strptime("2026-02-01", "%Y-%m-%d").date()

    excluded = set()  # or your excluded set of date objects

    current = start_date
    while current <= end_date:
        tick_start = time.monotonic()

        if current not in excluded:
            main(current.strftime("%Y-%m-%d"))

        current += timedelta(days=1)

        elapsed = time.monotonic() - tick_start
        sleep_s = max(0.0, 86400.0 - elapsed)
        print(f"[{datetime.now():%F %T}] Sleeping {sleep_s:.1f}s until next date")
        time.sleep(sleep_s)

