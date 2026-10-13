"""Full-resolution dynamic spectra from slow-buffer ``.ms.tar`` data.

Runs ``dynspec_map_v3`` on every 10 s slow-buffer integration of the chosen
UTC hours, for all 16 subbands, and reduces them with ``dynspec_reduce_v3``
into the usual layout (``<out-dir>/<baseline>/<DATE>-<corr>.fits``,
3072 channels × N integrations, no frequency averaging).

Map tasks are pinned to the given per-node Celery queues (weighted by the
worker concurrency), so nothing lands on nodes that are busy with other work.

Before submitting, one file per subband is extracted on local NVMe and checked:
ROW_NUMS rows really are the named baselines, the channel frequencies match
the FITS axis, and the integration time.  The first integration's TIME column
sets the FITS time axis.

Example (as the pipeline user on a calim node)::

    python pipeline/produce_dynspec_slow.py --date 2026-10-13 --hours 08 09 \\
        --queues calim05:20 calim06:8
"""
import argparse
import os
import re
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from casacore.tables import table
from celery import group

from orca.transform.spectrum_v3 import (
    ROW_NUMS, N_CHAN_OUT, _CHAN_WIDTH_MHZ, _FREQ0_MHZ,
    _baseline_names, _extract_ms_tar, dynspec_map_v3, dynspec_reduce_v3,
)

FREQS = [
    "13MHz", "18MHz", "23MHz", "27MHz", "32MHz", "36MHz",
    "41MHz", "46MHz", "50MHz", "55MHz", "59MHz", "64MHz",
    "69MHz", "73MHz", "78MHz", "82MHz",
]
MJD_EPOCH = datetime(1858, 11, 17)
FILE_RE = re.compile(r"(\d{8}_\d{6})_(\d+MHz)\.ms\.tar$")


def file_time(path: Path) -> datetime:
    """Start time from '20261013_080001_13MHz.ms.tar' (seconds are truncated)."""
    return datetime.strptime(FILE_RE.search(path.name).group(1), "%Y%m%d_%H%M%S")


def list_files(root: Path, date: str, hours) -> dict:
    files = {}
    for spw in FREQS:
        found = []
        for hr in hours:
            found += sorted((root / spw / date / hr).glob(f"*_{spw}.ms.tar"))
        files[spw] = found
    return files


def preflight(files: dict, work_dir: str) -> tuple:
    """Check the first file of every subband; return (t0_centre, inttime_s, rows)."""
    t0, inttime, rows_seen = None, None, set()
    for sb, spw in enumerate(FREQS):
        if not files[spw]:
            print(f"  {spw}: no files")
            continue
        tmp_dir, ms = _extract_ms_tar(str(files[spw][0]), work_dir)
        try:
            with table(ms, ack=False) as t:
                tcross = t.query("ANTENNA1!=ANTENNA2")
                a1, a2 = tcross.getcol("ANTENNA1"), tcross.getcol("ANTENNA2")
                tcross.close()
                times = np.unique(t.getcol("TIME"))
                interval = float(t.getcol("INTERVAL")[0])
            with table(f"{ms}/ANTENNA", ack=False) as ta:
                names = list(ta.getcol("NAME"))
            with table(f"{ms}/SPECTRAL_WINDOW", ack=False) as ts:
                freqs = ts.getcol("CHAN_FREQ")[0] / 1e6
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        if len(times) != 1:
            sys.exit(f"{spw}: expected one integration per file, got {len(times)}")
        expect_f = _FREQ0_MHZ + _CHAN_WIDTH_MHZ * (sb * N_CHAN_OUT + np.arange(N_CHAN_OUT))
        if len(freqs) != N_CHAN_OUT or np.max(np.abs(freqs - expect_f)) > 1e-6:
            sys.exit(f"{spw}: channel frequencies do not match the FITS axis")
        for name, row in ROW_NUMS:
            got = sorted([names[a1[row]], names[a2[row]]])
            ok = got == _baseline_names(name)
            rows_seen.add((name, row, ok))
            if not ok:
                print(f"  WARNING {spw}: row {row} is {got}, not {name}; map will look it up by name")
        t_centre = MJD_EPOCH + timedelta(seconds=float(times[0]))
        if t0 is None or t_centre < t0:
            t0 = t_centre
        inttime = interval
        print(f"  {spw}: {len(files[spw])} files, first {t_centre.isoformat()} UTC, "
              f"int {interval:.4f} s, {freqs[0]:.4f}-{freqs[-1]:.4f} MHz, rows OK")
    return t0, inttime, rows_seen


def parse_queues(specs):
    """['calim05:20', 'calim06:8'] -> [('calim05', 20), ('calim06', 8)]"""
    out = []
    for s in specs:
        q, _, w = s.partition(":")
        out.append((q, int(w or 1)))
    return out


def assign_queues(n_tasks: int, queues):
    """Weighted interleaved queue assignment (e.g. 20:8 → 5 of every 7 go to the first)."""
    total = sum(w for _, w in queues)
    credit = {q: 0.0 for q, _ in queues}
    out = []
    for _ in range(n_tasks):
        for q, w in queues:
            credit[q] += w / total
        q = max(credit, key=credit.get)
        credit[q] -= 1.0
        out.append(q)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--date", required=True, help="UTC date YYYY-MM-DD")
    p.add_argument("--hours", nargs="+", required=True, help="UTC hours, e.g. 08 09")
    p.add_argument("--root", default="/lustre/pipeline/slow")
    p.add_argument("--out-dir", default="/lustre/pipeline/dynspec")
    p.add_argument("--queues", nargs="+", required=True, help="queue[:weight], e.g. calim05:20 calim06:8")
    p.add_argument("--reduce-queue", help="queue for the reducer (default: first --queues entry)")
    p.add_argument("--use-ms-flags", action="store_true", help="NaN out MS-flagged samples")
    p.add_argument("--work-dir", default="/fast/pipeline/dynspec_tmp")
    p.add_argument("--max-scans", type=int, help="only the first N integrations (for testing)")
    p.add_argument("--dry-run", action="store_true", help="preflight only, submit nothing")
    p.add_argument("--overwrite", action="store_true", help="allow replacing existing FITS for DATE")
    args = p.parse_args()

    hours = [f"{int(h):02d}" for h in args.hours]
    queues = parse_queues(args.queues)
    reduce_queue = args.reduce_queue or queues[0][0]

    existing = [str(Path(args.out_dir) / n / f"{args.date}-XY.fits")
                for n in ["incoherent-sum"] + [n for n, _ in ROW_NUMS]]
    existing = [e for e in existing if os.path.exists(e)]
    if existing and not args.overwrite:
        sys.exit(f"Output exists (use --overwrite): {existing}")

    files = list_files(Path(args.root), args.date, hours)
    n_files = sum(len(v) for v in files.values())
    print(f"{args.date} hours {hours}: {n_files} files in {args.root}")
    if n_files == 0:
        sys.exit("nothing to do")

    print("Preflight (first file per subband):")
    t0, inttime, _ = preflight(files, args.work_dir)

    # Timeline: file names truncate to whole seconds and drift by 0.031 s per
    # integration, so index by rounding (t - t_first) / INTERVAL.
    t_first = min(file_time(f) for v in files.values() for f in v)
    tasks = []
    for sb, spw in enumerate(FREQS):
        seen = {}
        for f in files[spw]:
            scan = int(round((file_time(f) - t_first).total_seconds() / inttime))
            if scan in seen:
                sys.exit(f"{spw}: {f.name} and {seen[scan].name} map to the same scan {scan}")
            seen[scan] = f
            tasks.append((sb, scan, str(f)))
    if args.max_scans:
        tasks = [t for t in tasks if t[1] < args.max_scans]
    n_scans = max(t[1] for t in tasks) + 1
    print(f"Timeline: {n_scans} integrations from {t0.isoformat()} UTC "
          f"(CDELT1 {inttime:.4f} s), {len(tasks)} map tasks")

    meta = {
        "DATE-END": ((t0 + timedelta(seconds=inttime * (n_scans - 1))).isoformat(),
                     "UTC centre of last integration"),
        "INTTIME": (inttime, "[s] integration time"),
        "DATASRC": (args.root, "slow-buffer .ms.tar, full resolution"),
        "UTHOURS": (",".join(hours), "UTC hours processed"),
        "BCAL": ("none", "no bandpass applied"),
        "MSFLAGS": (bool(args.use_ms_flags), "MS FLAG column applied"),
        "ORIGIN": "orca spectrum_v3 / produce_dynspec_slow.py",
        "DATE": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # Interleave subbands in time order so every subband progresses together.
    tasks.sort(key=lambda t: (t[1], t[0]))
    qs = assign_queues(len(tasks), queues)
    for q, _ in queues:
        print(f"  {q}: {qs.count(q)} tasks")
    if args.dry_run:
        print("Dry run: nothing submitted.")
        return

    sigs = [
        dynspec_map_v3.si(sb, scan, ms, None, args.use_ms_flags).set(queue=q)
        for (sb, scan, ms), q in zip(tasks, qs)
    ]
    t_start = datetime.now()
    res = group(sigs).apply_async()
    print(f"[{t_start:%F %T}] Submitted map group {res.id}", flush=True)

    raw = res.get(disable_sync_subtasks=False, propagate=False)
    spectra = [r for r in raw if not isinstance(r, Exception)]
    n_failed = len(raw) - len(spectra)
    print(f"[{datetime.now():%F %T}] Maps done: {len(spectra)}/{len(raw)} OK, {n_failed} failed", flush=True)
    if n_failed:
        for (sb, scan, ms), r in zip(tasks, raw):
            if isinstance(r, Exception):
                print(f"  FAILED {ms}: {r!r}")
    meta["NFAILED"] = (n_failed, "map tasks that failed (zero-filled)")

    red = dynspec_reduce_v3.si(spectra, t0, args.out_dir, meta).set(queue=reduce_queue).apply_async()
    print(f"[{datetime.now():%F %T}] Submitted reduce {red.id} on {reduce_queue}", flush=True)
    red.get(disable_sync_subtasks=False)
    print(f"[{datetime.now():%F %T}] Done in {datetime.now() - t_start}. "
          f"FITS in {args.out_dir}/<baseline>/{args.date}-<corr>.fits")


if __name__ == "__main__":
    main()
