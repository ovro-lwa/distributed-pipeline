"""Dynamic spectrum production pipeline.

This script orchestrates dynamic spectrum extraction across multiple dates
and frequencies using Celery distributed tasks. Excludes problematic dates
and generates waterfall plots.
"""
from pathlib import Path
from datetime import datetime
from celery import group
from orca.celery import app                

from orca.transform.spectrum_v2 import dynspec_reduce_v2
from orca.transform.spectrum_v2 import dynspec_map_v2
from datetime import datetime, timedelta

# Your excluded dates
excluded_dates = set(["2025-03-16","2025-03-17","2025-03-23","2025-03-24",
    "2025-04-24", "2025-04-25", "2025-04-26", "2025-04-27",
    "2025-04-28", "2025-04-29", "2025-04-30", "2025-04-13",
    "2025-04-12", "2025-04-11", "2025-04-10","2025-04-14","2025-04-15","2025-04-16","2025-04-17",
    "2025-04-18","2025-04-19","2025-04-20","2025-04-21","2025-04-22","2025-04-23",
])

# Convert to datetime.date objects for easy comparison
excluded = {datetime.strptime(d, "%Y-%m-%d").date() for d in excluded_dates}

# Define range
start_date = datetime.strptime("2025-11-01","%Y-%m-%d").date()
end_date = datetime.strptime("2025-11-03", "%Y-%m-%d").date()

# Generate allowed dates
allowed_dates = [
    d.strftime("%Y-%m-%d")
    for i in range((end_date - start_date).days)
    if (d := start_date + timedelta(days=i)) not in excluded
]

# Result



# ─────────────────────────── User-configurable parameters ────────────────
ROOT   = Path("/lustre/pipeline/night-time/averaged")
DATES   = allowed_dates
#["2025-04-24","2025-04-25", "2025-04-26","2025-04-27","2025-04-28", "2025-04-29", "2025-04-30","2025-04-13", "2025-04-12", "2025-04-11", "2025-04-10"]
HOURS  = [f"{h:02d}" for h in range(5, 13)]         
FREQS  = [
    "13MHz","18MHz","23MHz","27MHz","32MHz","36MHz",
    "41MHz","46MHz","50MHz","55MHz","59MHz","64MHz",
    "69MHz","73MHz","78MHz","82MHz",
]
OUT_DIR = "/lustre/pipeline/dynspec"
USE_MS_FLAG = False
# ──────────────────────────────────────────────────────────────────────────

def ts_from_ms(ms_path: Path) -> str:
    """
    Extract 'YYYYMMDD_HHMMSS' time-stamp from
      20250419_010007_73MHz_averaged.ms  →  '20250419_010007'
    """
    return ms_path.stem.split('_', 2)[0] + '_' + ms_path.stem.split('_', 2)[1]


def main(DATE: str) -> None:
    t0 = datetime.now()
    print(f"[{t0:%F %T}] Building global timeline …")

    # 1. Collect every unique time-stamp across ALL sub-bands 
    all_ts: set[str] = set()
    for spw in FREQS:
        for hr in HOURS:
            for ms in (ROOT / spw / DATE / hr).glob("*.ms"):
                all_ts.add(ts_from_ms(ms))

    ts_list  = sorted(all_ts)                   # ordered timeline
    ts_index = {ts: i for i, ts in enumerate(ts_list)}
    print(f"Timeline contains {len(ts_list)} distinct scans")

    # 2. Build Celery signatures for map tasks 
    #map_sigs = []
    #for sb, spw in enumerate(FREQS):
    #    for hr in HOURS:
    #        for ms in (ROOT / spw / DATE / hr).glob("*.ms"):
    #            scan_no = ts_index[ts_from_ms(ms)]
    #            map_sigs.append(
    #                app.signature(                
    #                    "orca.transform.spectrum_v2.dynspec_map_v2",
    #                    args=(sb, scan_no, str(ms), None),
    #                    kwargs={"use_ms_flags": USE_MS_FLAG} 
    #                )
    #            )
    map_sigs = []
    for sb, spw in enumerate(FREQS):
        for hr in HOURS:
            for ms in (ROOT / spw / DATE / hr).glob("*.ms"):
                scan_no = ts_index[ts_from_ms(ms)]
                map_sigs.append(
                    dynspec_map_v2.si(                         # <- immutable, kwargs-only
                        subband_no=sb,
                        scan_no=scan_no,
                        ms=str(ms),
                        bcal=None,
                        use_ms_flags=USE_MS_FLAG,
                    )
                )
    
    print(f"Created {len(map_sigs)} map signatures")

    # 3. Submit maps as one group 
    map_group  = group(map_sigs)
    map_result = map_group.apply_async()
    print(f"[{datetime.now():%F %T}] Submitted map group: {map_result.id}")

    # 4. Wait for maps & gather results (list of lists of dicts) 
    #spectra_lists = map_result.get(disable_sync_subtasks=False)
    spectra_lists_raw = map_result.get(disable_sync_subtasks=False, propagate=False)
    num_total = len(spectra_lists_raw)
    num_failed = sum(isinstance(r, Exception) for r in spectra_lists_raw)
    num_success = num_total - num_failed

    # Filter out failed
    spectra_lists = [r for r in spectra_lists_raw if not isinstance(r, Exception)]

    print(f"[{datetime.now():%F %T}] Maps complete — {num_success}/{num_total} succeeded, {num_failed} failed")


    #print(f"[{datetime.now():%F %T}] Maps complete — {len(spectra_lists)} snapshots")

    # 5. Launch reducer 
    start_ts = datetime.fromisoformat(f"{DATE}T{HOURS[0]}:00:00")
    reduce_result = dynspec_reduce_v2.apply_async(
        args=(spectra_lists, start_ts, OUT_DIR)
    )
    print(f"[{datetime.now():%F %T}] Submitted reduce task: {reduce_result.id}")

    reduce_result.get()            # blocks until done
    t1 = datetime.now()

    print(f"[{t1:%F %T}] All done!  FITS cubes in: {OUT_DIR}")
    print(f"Total elapsed: {t1 - t0}")

if __name__ == "__main__":
    for DATE in DATES:
        main(DATE)

