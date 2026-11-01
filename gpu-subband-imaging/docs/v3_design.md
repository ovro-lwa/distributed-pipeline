# gsi v3: in-memory band-hour pipeline (design)

## 0. Sources

- **Pipeline code:** `/private/tmp/claude-501/-Users-nikita-caltech-courses-distributed-pipeline/38f3812d-2c74-4a36-90ed-8f2853117c8f/scratchpad/e2e/code_new/gpu-subband-imaging`. This is gsi-v2 13405ce plus the test changes.
- **Peeler:** `~/caltech_courses/ttcal/julia_ttcal/TTCalX-sponge` (branch `feature/sponge-restore-fast-zest`, 7ca3845, TTCalX 0.3.0).
- **Imager:** `~/caltech_courses/ttcal/julia_ttcal/TTCalX-imgspeed/python/ttcalx_imager/src/ttcalx_imager` (86afd89). The e2e run uses `/lustre/nkosogor/peel_exp/imgspeed/combined/TTCalX/python/ttcalx_imager/src`.
- **Review:** `/private/tmp/claude-501/-Users-nikita-caltech-courses-distributed-pipeline/38f3812d-2c74-4a36-90ed-8f2853117c8f/scratchpad/report/body.tex`, sections "Measured performance" and `sec:future`.
- **Fresh measurements:** the e2e runs of 2026-10-31, LST 05h, 23 MHz, "new" code. Logs are under `/lustre/nkosogor/peel_exp/e2e/out_new{,_c00}`.
- **Kept-sample counts:** read from the hour summaries `QA/<band>MHz-summary.json` of the production nights in `/lustre/pipeline/snapshots_gpu_pipeline`.

## 1. Measured data sizes

### 1.1 One averaged snapshot MS

Read-only casacore on `kept_prod_c00/23MHz/2026-10-31/05h/20261031_101709_23MHz.ms`:

| item | value |
|---|---|
| rows | 62 128 = 61 776 cross (352·351/2) + 352 autos, one time step |
| antennas | 352 |
| channels | 48 (192 / chanbin 4), 95.703 kHz each; 22.62–27.12 MHz at 23 MHz |
| correlations | 4 (XX, XY, YX, YY) |
| DATA | complex64 (48, 4) per row. No CORRECTED_DATA, MODEL_DATA or WEIGHT_SPECTRUM |
| FLAG | bool (48, 4) per row |
| WEIGHT / SIGMA | float32 (4,) per row |
| UVW | float64 (3,) per row |
| on disk | 103.7 MB, **102 inodes** (DATA TSM file 96.4 MB) |
| flag fraction | 0.62 for this 23 MHz file on 10-31 |

In-memory sizes:

| array | bytes per snapshot |
|---|---|
| DATA (62 128 × 48 × 4 × complex64) | 95.4 MB |
| FLAG as bool (row, chan, pol) | 11.9 MB |
| FLAG as bool (row, chan) | 3.0 MB |
| UVW | 1.5 MB |
| WEIGHT | 1.0 MB |
| **full-pol snapshot ("slot" form)** | **≈ 100 MB** |

The (row, chan) flag form is exact after peeling. `write_ms_fast!` writes the polarisation-ORed baseline flag to all four correlations (`TTCalX-sponge/src/gpu/ms_io.jl:164`).

### 1.2 What the hour imager actually needs

`ttcalx_imager.msio.load_samples` keeps only the samples with XX and YY unflagged, as I and V. From the hour summaries (`products[...][0].npts / n_snap`):

| case | kept samples per snapshot | fraction of 2.965 M |
|---|---|---|
| 18–78 MHz, nights 10-29/10-30 (00h–02h) | 2.01–2.03 M | 68 % |
| 82 MHz (85–100 MHz flagged) | 1.22 M | 41 % |
| e2e 23 MHz, 10-31 | 1.10 M | 37 % |

Deep products hold about 700 M samples per band-hour, typically 358 snapshots.

Per-snapshot cost of each storage layout:

| layout | per snapshot | per band-hour (358) |
|---|---|---|
| v2 hour job host layout (`load_samples`: u, v, w, nw f32 + I, V complex64 = 32 B/sample) | 64.8 MB | 23.2 GB |
| v3 compact (I, V complex64 per kept sample = 16 B; per row: uvw f64, natw f32, ok bitmask = 34 B) | 32.4 + 2.1 = **34.5 MB** | **12.4 GB** typical, 7.6 GB at 82 MHz, 17.7 GB worst case (no flags) |
| full-pol (4 corr + flags + weights) | 100 MB | 35.8 GB |

The compact u, v, w are rebuilt as `(uvw[:, k:k+1] * freqs/C)[ok].astype(float32)`. That is one float64 multiply and then a cast, so the values are bit-identical to `load_samples`, on numpy or CuPy.

The v2 hour job holds the 23.2 GB of samples in RAM. With `TTCALX_CHUNK_CACHE=auto` it also holds host-side concatenations for chunks that do not fit on the GPU, about 24 B/sample more. Its peak RSS is likely 25–40 GB. This is not measured yet; step 0 measures it.

### 1.3 Node state and the current per-snapshot flow

calim07 at the time of reading:
- RAM: 503 GB total, **66 GB available**, swap full (3/3 GB).
- Disk: `/fast` 127 GB free (97 % used); `/` (`/tmp`) 48 GB free.
- GPUs: 2 × RTX A4000.
- Lustre quota: 1 899 636 of 2 000 000 files, so about **100 k inodes free**.
- One band-hour of kept MSes is 358 × 102 ≈ **36.5 k inodes**.

The current flow goes through disk at every step (body.tex `sec:future`): untar → flag_cal putcol → mstransform → zest read/write → chgcentre copy → imager read → keep_ms to Lustre → hour re-read. That is about 2.4 GB of local I/O and 0.63 GB of Lustre I/O per snapshot.

## 2. Bottleneck numbers used below

### 2.1 Night 2026-10-28 medians (body.tex)

| band | zest (read / peel / write) | img_gpu | s/file | hour job |
|---|---|---|---|---|
| 18 | 3.34 (0.86 / 1.94 / 0.47) | 0.73 | 3.68 | 505 s |
| 55 | 4.21 (0.96 / 2.65 / 0.50) | 2.16 | 4.45 | 752 s |
| 82 | 4.83 (0.98 / 3.24 / 0.52) | 3.51 | 5.10 | 926 s |

- Prep is 1.3 + 3.2 + 4.9 = 9.4 s CPU per file; over 4 feeders that is 2.35 s/file per GPU.
- Hour job load + rotation is 50 / 60 / 84 s with the GPU idle. Deconvolution is about 60 % of the hour job.
- At 82 MHz the peel and imager processes together use 6.75 kernel-seconds in 5.1 s of wall time, so they time-slice the GPU.

### 2.2 e2e 2026-10-31, 23 MHz, LST 05h (new peeler and imager)

| phase (mean s) | calim00 (RAM rich) | calim07 (RAM poor) |
|---|---|---|
| stage_ms | 1.35 | 6.81 |
| flag_cal / average | 3.13 / 4.43 | 9.56 / 14.33 |
| zest (read / peel / write) | 3.10 (0.94 / 1.63 / 0.50) | 9.60 (**5.59** / 1.99 / 1.41) |
| chgcentre | 0.58 | 4.33 |
| image (gpu) | 1.61 (0.56) | 7.32 (0.76) |
| keep_ms (Lustre) | 2.59 | 7.30 |
| batch | 1183 s / 356 = **3.32 s/file** | 3745 s / 353 = **10.6 s/file** |
| hour job | 491 s (load 46, deconv 243, movie 14) | 484 s (load **171**, deconv ~144, movie 41) |

On calim07 the MS reads miss the page cache because of memory pressure (zest read 5.6 s), and prep costs about 31 s CPU per file. This is the case v3 helps most.

## 3. Architecture

### 3.1 Process topology (one batch = one band × one LST hour = about 358 snapshots, one GPU)

```
 worker.py (conda py38, parent; pinned to cores 0-23 as today)
  |
  |-- prep ProcessPool (spawn, max_feeders)             CPU
  |     tar (Lustre) --stream--> /dev/shm raw MS --casacore--> numpy cal + avg + fix_uvw
  |     --> writes a SLOT in the shm ring                (no MS written)
  |
  |-- SlotRing (/dev/shm/gsi3_<pid>_<k>, ~100 MB each, lookahead+2 slots, reused)
  |
  |-- zest_daemon.jl (Julia, existing daemon)           GPU, serial
  |     "SHM <slot>"  -> mmap slot, peel (zest_sky_rfi!), write back in place
  |
  |-- gsi_v3_engine.py (imager venv python)             GPU + 2 host threads
        "image_slot": gpu_select(slot) -> rotate to LST centre in memory -> CLEAN I + dirty V -> FITS
        store thread: slot -> compact I/V record -> HourStore (RAM | spill) [+ checkpoint/archive]
        "hour": close of the peel daemon, then image_hour_snaps(HourStore) -> 1 h + 10 min products
```

Why this split:
- **Prep in processes, not threads.** python-casacore holds the GIL while reading; `snapload.py` says so and already moves its reader to a process. v2 `prep: numpy` runs `fastcal.flag_calibrate` in threads.
- **Peeler stays a separate Julia process, fed through shared memory.** This keeps the daemon's crash isolation and recycle logic, and needs no TTCalX change. TTCalX hard-depends on PyCall, which conflicts with juliacall in one process. juliacall is an optional later step (step 8).
- **The engine runs the imager venv** (CuPy, cuFINUFFT, python-casacore). The parent stays in the conda env (CASA, `flag.badants` through the dev env), so no environment merge is needed.
- **The slot layout equals the imager's "raw" reader layout:** `small` dict + `raw = {DATA: (nrow, 48, 4) c64, FLAG: (nrow, 48, 4) bool}`. This is the layout of `snapload._raw_to_shm`, so the engine calls the existing `snapshot.gpu_select(small, raw, ...)` and `SnapshotImager._compute` unchanged.

### 3.2 Slot format (`src/gpu_subband_imaging/v3/slots.py`, mirrored in `workers/zest_daemon.jl`)

- **Header (4096 B):** magic `GSISLOT1`, a u32 JSON length, then JSON. The JSON holds: `nrow, nchan, ncorr`; offsets of `a1` (i32), `a2` (i32), `uvw` (f64 (nrow, 3)), `weight` (f32 (nrow, 4)), `data` (c64 (nrow, nchan, 4)), `flag` (bool (nrow, nchan, 4)); `time, tstart, tmid, time_centroid, mjd`, with `mjd = (min TIME + max TIME)/2/86400` as `ms_mjd`; `phase_dir, freqs, chan_width, ant_pos, telescope, observer, field_name, stem`; and `state` (PREPPED, PEELED, FREE).
- Arrays are C-order and 64-byte aligned. Julia maps them with reversed dims, e.g. `Mmap.mmap(io, Array{ComplexF32,3}, (4, nchan, nrow), off_data)`.

Julia side, a new `"SHM <path>"` branch in the daemon loop:
- **`read_slot(path; gpu)`:** gathers the cross rows exactly as `read_ms_fast` does (lo/hi antennas, unique check, flags `any` over pols, ComplexF32 → ComplexF64 upload). It builds `GPUVisibilities`, `GPUMetadata` and `GPUCalibration` as `read_ms_fast` does (`ms_io.jl:133`). `channels`, `ant_pos` and `ra/dec` come from the JSON, not from `_ms_subtables`.
- **`write_slot!(path, vis, rows)`:** scatter as in `write_ms_fast!`. Cross-row DATA becomes `ComplexF32.(vis)`, which rounds the same way as casacore's double→float putcol. FLAG of all four pols becomes the baseline flag; autos are untouched. The result is byte-identical to v2's write followed by the next process's read.
- The `ZESTED` / `FAILED` replies are unchanged, so `ZestDaemon.zest()` only needs `zest_slot(path)`.

### 3.3 Snapshot imaging from a slot (engine `image_slot`)

1. `small`/`raw` views come from the slot (zero copy).
2. `snapshot_phase_centre: lst` currently copies the MS and runs chgcentre (`worker.py:246`). Instead, compute `R = rotate.snapshot_rotation(uvw_cross, a1, a2, ant_pos, time, ra_lst, dec_lst)` once and apply `uvw' = R·uvw` and `vis' = vis·exp(2πi(w − w')/λ)` on the GPU before `gpu_select`. This is chgcentre-exact (rotate.py). The **same R** goes into the hour record, because the hour job uses the same LST-hour centre.
3. `SnapshotImager._compute` / `_finish` write `<prefix>-I-image.fits` and `-V-image.fits`. The parent keeps `fpack`, frame render, `v_stats` and the archive move unchanged (`Worker._image_archive` without the chgcentre/keep_ms branches).

### 3.4 Hour accumulation and hour products

The store thread in the engine turns each PEELED slot into a compact record:
- It reuses the imager's own expressions so the values are identical by construction. Refactor `msio.read_ms` into `msdata_from_arrays(...)` and `load_samples` into `samples_from_msdata(ms, ra, dec, compact=True)`.
- `ok = ~(flag[:, :, 0] | flag[:, :, 3])` on the cross rows, `I = (0.5*(XX+YY)).astype(c64)[ok]`, `V = (0.5j*(YX−XY)).astype(c64)[ok]`.
- Per row it keeps `uvw` (f64), `natw = 0.5*(W_XX + W_YY)` (f32) and the `ok` bitmask.
- Metadata: `R`, `time`, `tstart`, `freqs`, `chan_width`, `pd`, `zero_frac`, `flagfrac`, `stem`.

Then the slot is released.

**Can the hour be streamed or done incrementally? No for the products, yes for some setup.**
- **Briggs/Tukey weights are global per interval.** `WeightGrid` uses the density over all accepted samples of the interval (`hour.py:79-104`). A sample's imaging weight is known only when the interval is complete. Interval membership (`bnd = floor(k·nfull/nint)`) depends on the final snapshot count, including rejected ones.
- **Dirty, PSF and every major cycle grid the weighted samples again.** The residual is `grid(iw·(vis − degrid(model)))` (`HourGrid.image`, `WSnap.image`). The multiscale CLEAN with WSClean major cycles needs **all samples resident** in every cycle. A grid-once, image-domain-only deconvolution would not be equivalent to v2/WSClean. It could be a quick-look option, never the default.
- **What is incremental:**
  - R per snapshot (casacore measures, about 0.1 s);
  - the sample compaction;
  - wmax and uvmax;
  - the pilot V-QA statistics (already per snapshot);
  - optionally the density grid of the deep interval (saves 5–10 s per product; low priority, step 9).

So v3 keeps all compact samples of the band-hour in RAM (or spilled, section 5). At the end of the batch:

1. The parent closes the zest daemon, which frees its 4 GiB GPU soft limit. The hour stage then has the whole 16 GB.
2. The parent runs QA: `hour_lst.write_qa` from the in-memory `self._qa` rows, and still writes `.snapshot_stats/*.jsonl`.
3. The parent sends `{"op": "hour", products, lst_hour, exclude_stems}` to the engine. Stems replace indices, which also fixes body.tex P5.
4. The engine calls a new `ttcalx_imager.hour.image_hour_snaps(snaps, band, out_prefix, ra, dec, products, exclude=...)`. This is `image_hour` from line 335 on, split from the load step, so the CLI `image_hour` = `load_all` + `image_hour_snaps` stays unchanged.
   - `snaps` are `LazySnap` mappings over the store. `I` and `V` are views; `u`, `v`, `w`, `nw` are expanded per chunk.
   - Add `HourGrid._n(i)` (record `n`) instead of `len(s["u"])`, so a length check does not expand arrays.
   - Add `TTCALX_CHUNK_CACHE=store`, which expands u, v, w on the GPU from per-row uvw. This avoids the host-cache duplicate (about 24 B/sample).
5. The 1-h and 10-min products come from the same `snaps` list. This is exactly what `image_hour` does after its load, and nothing is re-read.
6. The parent runs `hour_lst.place_products`, provenance, and `movie.stitch`. The movie runs in a thread started when the last frame is written, so it overlaps the GPU hour stage (14–41 s hidden).

## 4. What still goes to disk

**Always, as v2:**
- snapshot FITS (I CLEAN in `snapshots_clean`, V in `snapshots`, fpacked);
- PNG frames and the movie;
- `QA/` (csv, png, `pilot_qa_rejected.txt`, `.snapshot_stats`);
- the hour products in `I|V/deep|10min`;
- `provenance.json`;
- worker logs with the same `timing ... phase=` lines (so `night_report.py` and `e2e_compare.py` keep working).

**New, small:**
- `QA/<band>MHz-peel.jsonl`: one line per snapshot with the `PEEL_SOLVE` / `PEEL_SUMMARY` / `PEEL_RESTORE` content (today only in logs);
- provenance additions: `v3` block (modes, store backend, spill path, archive path + sha256, n_records, peak RSS, MemAvailable at start and end).

**Optional archive** (`v3.archive: none | iv | full`, default none): **one record file per band-hour** plus an index:
`<band dir>/archive/<band>MHz-<date>-<LST>h-peeled.gsiv` and `.gsiv.idx.jsonl`.

That is 2 inodes instead of about 36.5 k, which solves the quota problem.

| mode | content | per band-hour | per night (165 band-hours) |
|---|---|---|---|
| `iv` | the compact hour records (enough to redo every hour product and every snapshot image) | 12.4 GB | about 2 TB |
| `full` | full-pol complex64 + flags + weights | 36 GB | — |

For comparison, the kept MSes are about 6.5 TB/night written and 6.5 TB read back. `full` allows `gsi export-ms <file.gsiv> --stamp S --template <raw tar>`, which rebuilds a CASA MS for orca tools.

**GSIV v1 format** (`src/gpu_subband_imaging/v3/hourstore.py`; reader duplicated in `ttcalx_imager/gsiv.py`):
- File header: magic + JSON (band, date, LST, a1/a2, ant_pos, freqs, chan_width, mode, layout).
- Append-only records: 64-B record header, 64-B aligned arrays, crc32.
- The `.idx.jsonl` line (stamp, offset, nbytes, crc, n_kept, time, tstart, R, zero_frac, flag_frac, peel summary) is written **after** the record bytes are flushed and fsynced. A torn tail is detected and truncated.
- Why not HDF5: no crash-safe append without SWMR, and h5py is not guaranteed in both envs. With this format `np.memmap` reads directly.

**Checkpoint and restart** (`v3.checkpoint: none | local | archive`, default local when `hour_source: memory`):
- **Where:** the store thread appends every record to `<spill_dir>/<run_key>/<band>MHz_<LST>.gsiv` (local NVMe). With `checkpoint: archive` it writes to the Lustre archive path instead. Cost is about 35 MB per snapshot, roughly 0.03 s on NVMe, in a background thread. A RAM-mode store and its checkpoint hold the same bytes.
- **Warm start:** `zest_daemon.jl` writes `prev_gains` after each snapshot to `<checkpoint>.warm` (Serialization, atomic rename) and loads it at start. `_warm_usable` still enforces the 60 s age and channel checks, so a resumed batch peels the next snapshot as an uninterrupted run would.
- **Batch start:**
  1. If `provenance.json` has hour status ok, there is nothing to do.
  2. Open the checkpoint, validate the index and crcs, truncate garbage, load or memmap the valid records.
  3. For each stamp:
     - FITS and record present: skip.
     - Record present but FITS missing: re-image from the record (the snapshot imager uses only I, V, weights and the ok mask, all in the record).
     - FITS present but record missing: **replay** (prep + peel only, no imaging, about 2–3 s).
     - Neither: full path.
  4. If all records are present, go straight to the hour stage. This is the retry of a failed hour stage.
- **Retry placement:** the orchestrator prefers the band-hour's node for retries when `checkpoint: local`, the same affinity as `keep_ms_local` (`orchestrator._choose`). On another node, the stamps are replayed.
- **Cleanup:** after the hour products succeed, the local checkpoint is deleted (`v3.delete_checkpoint`, default true), as `delete_kept_ms` does today. If the archive is enabled, the checkpoint is copied to Lustre in a background thread during the hour stage (one sequential 12 GB write).

## 5. Memory budget and fallback

### 5.1 Per GPU process, typical band (2.03 M kept samples/snapshot, 358 snapshots)

| component | size |
|---|---|
| HourStore compact | 12.4 GB (7.6 GB at 82 MHz; 17.7 GB worst case) |
| prep: 4 feeders × (409 MB raw MS in tmpfs + about 1 GB numpy working arrays, 192-ch DATA is 381 MB) | about 5.6 GB peak |
| SlotRing: (lookahead 8 + 2) × 100 MB | 1.0 GB |
| Julia daemon RSS | about 3–4 GB (measure) |
| engine (CuPy, cuFINUFFT host side) + 2 × 1.3 GB pinned staging for hour chunks | about 5 GB |
| hour-stage transients (40 M-sample chunk expansion) | about 2 GB |
| **total** | **about 28–30 GB per process, about 60 GB per node with 2 GPUs** |

`v3.pin_store` defaults to false. Pinning 12 GB per process on a node like calim07 is risky, so only the staging buffers are pinned.

### 5.2 Node policy (`v3.store: ram | spill | auto`, default auto)

1. **At batch start:**
   - Estimate `need = n_todo × est_record_bytes(band) × 1.15 + fixed`, where `est_record_bytes` is 34.5 MB, or 21 MB for bands ≥ 82 MHz. Replace it with the first snapshot's real value.
   - Read `MemAvailable` from `/proc/meminfo`. Subtract `v3.ram_reserve_gb` (default 16) and the other GPU's reservation. Each process writes `/dev/shm/gsi3_mem_gpu<k>.json`, so two processes on one node do not both claim the same RAM.
   - If it fits: `ram`. Otherwise: `spill` to the first `v3.spill_dirs` entry with `free ≥ need × 1.2`. Defaults: `/fast/pipeline/gsi3_spill`, then `/tmp/gsi3_spill`. Never Lustre, unless `checkpoint: archive`, in which case the archive file is the spill.
2. **During the batch** (checked every 10 snapshots): if MemAvailable drops below the reserve, switch to spill (the records are already in the checkpoint file, so free the RAM copies and memmap), cut the lookahead to 2, and drop untar to `work_root` instead of `/dev/shm`. Log `V3_MEM_SPILL`.
3. **Spill cost:** about 0.03 s per snapshot of NVMe append. The hour stage reads 12.4 GB from page cache or NVMe in about 6–10 s, against v2's 46–171 s of casacore loading from Lustre.
4. **Expected outcomes:**
   - calim00 (about 340 GB available): `ram` on both GPUs. It could even hold full-pol.
   - calim07 (30–45 GB available): one RAM process plus one spilled process, or both spilled (about 16 GB RAM each). `/fast` has 127 GB free, enough for about 10 band-hours of compact records.

## 6. Configuration (`config.py`)

Add `@dataclass(frozen=True) class V3Params` and `PipelineConfig.v3: V3Params = field(default_factory=V3Params)`, loaded in `load_pipeline` from `d.get("v3", {})`. All defaults reproduce v2:

```yaml
v3:
  untar_to: ""            # step 1: "" = work_root as v2 | /dev/shm/gsi3 (stream the tar from Lustre)
  prep_inmem: off         # step 2: off | shadow (compute both, log diffs, use v2) | on
  peel_io: ms             # step 3: ms | shm
  snapshot_io: ms         # step 4: ms | shm
  hour_source: kept_ms    # step 5/6: kept_ms | archive | memory
  write_kept_ms: true     # keep writing kept MSes (v2 hour job / comparison) while testing
  archive: none           # none | iv | full
  archive_root: ""        # Lustre; checked by _validate_lustre_write_path
  store: auto             # ram | spill | auto
  spill_dirs: [/fast/pipeline/gsi3_spill, /tmp/gsi3_spill]
  ram_reserve_gb: 16
  checkpoint: local       # none | local | archive (used only when hour_source != kept_ms)
  delete_checkpoint: true
  lookahead: 8
  prep_procs: 4           # replaces max_feeders for the v3 prep pool
  peel_backend: daemon    # step 8: daemon | juliacall
```

Validation in `__post_init__`:
- `peel_io=shm` needs `prep_inmem=on` and `prep=numpy`;
- `snapshot_io=shm` needs `peel_io=shm` and `imaging.imager=ttcalx`;
- `hour_source=memory` needs `snapshot_io=shm`, `layout=lst` and `stream_hour_images`;
- `archive != none` needs `archive_root`;
- `untar_to` must not be on Lustre.

## 7. Implementation plan (small steps, each switchable and each with a v2 equivalence test)

### Common test harness (step 0)

- **`scripts/v3_compare.py`** generalises `/lustre/nkosogor/peel_exp/e2e/e2e_compare.py` to roots `REF` (v2), `TEST` (v3 step) and `FLOOR` (a v2 rerun). It reuses `compare_pair`, `src_table`, `snapshots(step=10)` and `hour_products` including 10 min. It measures:
  - noise (MAD σ in horizon mask 0.40);
  - diff rms and max in σ;
  - peaks of **3C123, 3C134, 3C147, 3C196, 3C48, 3C295, 3C380**;
  - residual rms at the peeled **CasA, CygA, TauA, VirA, PerB, HerA, HyaA**;
  - **Jupiter** peak (restored below 41 MHz);
  - per-interval 10-min noise;
  - snapshot count, the QA-rejected set and the hour `n_loaded`.
- **`scripts/v3_vis_compare.py`** compares peeled visibilities: v2 kept-MS DATA/FLAG/WEIGHT/UVW against the v3 slot, record or MS, as bitwise equal, max |Δ| and relative rms.
- **Pass criteria:**
  - "bitwise" steps: visibilities byte-identical, images within the v2 rerun floor (cuFINUFFT atomics give 0.06–0.10 σ, body.tex);
  - "numeric" steps: diff rms ≤ max(1.5 × floor, 0.05 σ), diff max ≤ max(1.5 × floor, 0.5 σ);
  - noise ratio within ±1 %;
  - check-source peaks above 10 σ within max(1 %, 0.1 σ);
  - peeled residual rms within 5 % or 0.1 σ;
  - Jupiter within 2 %;
  - same QA rejections.
- **Data:** LST 05h, 2026-10-31, bands 23, 55, 82 (one per pixel tier), on calim00 and calim07, as in the e2e. A smoke set uses `max_files_per_hour: 24`.
- **Unit tests:** `tests/test_v3_*.py`, synthetic and runnable without GPU or CASA where possible.

### Step 0: baseline and instrumentation (no behaviour change)

- Log peak RSS and MemAvailable per batch and hour job (`resource.getrusage`, `/proc/meminfo`) in `worker.run_batch` and `hour_lst.run_band`.
- Run v2 twice: this gives the floor, and checks the report's claim that the peel is deterministic (md5 of DATA from peeling two copies of one MS).
- Speedup: 0.

### Step 1: untar into tmpfs, streamed from Lustre (`v3.untar_to`)

- `stages/stage_ms.py`: `stage(input, work_dir, untar_to=None)`. Use `tarfile.open(path, "r|*")` straight from Lustre into `/dev/shm/gsi3/<stamp>` and skip the local tar copy. Keep the 3-attempt retry and a size/member check.
- The worker deletes the raw MS from tmpfs right after averaging (it already does `shutil.rmtree(ms)`).
- **Test:** the averaged MS is byte-identical to v2 (`v3_vis_compare` bitwise), and so are the images.
- **Speedup:** stage 1.35 → about 0.7 s (c00) and 6.8 → about 2 s (calim07) of CPU per file. Wall time is unchanged while zest-bound (c00: zest 3.1 against 2.25 s/file prep). On calim07 this frees about 1.2 s/file of the 4-feeder budget.

### Step 2: in-memory numpy prep (`v3.prep_inmem: shadow` then `on`)

- New `src/gpu_subband_imaging/v3/fastprep.py`:
  - `prep_arrays(raw_ms, bp, xy, badants, chanbin, fix_uvw_mode) -> SnapArrays`: one casacore read; the `fastcal.flag_calibrate` math without the putcols; `average_like_mstransform(data, flag, weight, chanbin)`, a flag-aware weighted mean that reproduces mstransform's DATA, FLAG, WEIGHT and SIGMA including all-flagged bins;
  - `stages/uvw.py` refactored into `fix_uvw_arrays(...)`, with `fix_uvw(ms)` calling it so v2 behaviour is unchanged;
  - `quality.visibility_stats_arrays(...)`.
- `fastprep.write_ms_like(template_avg_ms, raw_ms, snap, out)` covers the paths that still need an MS (peel_io=ms, write_kept_ms). The template is the first file's mstransform output for the batch (one CASA call). Per snapshot: copy the template main table, SPECTRAL_WINDOW, DATA_DESCRIPTION and POLARIZATION; copy every other subtable (FIELD, OBSERVATION, POINTING, SOURCE, HISTORY, ...) from that snapshot's raw MS; putcol the main columns.
- The prep runs in a `ProcessPoolExecutor(spawn)` (`v3.prep_procs`), not threads. The GIL matters here.
- **Test:**
  - `shadow` on 3 bands × 24 files: max |ΔDATA| / rms ≤ 1e-6 (float32 level), FLAG identical, WEIGHT and SIGMA identical, UVW identical;
  - `casacore tablestructure` of `write_ms_like` equal to mstransform's, plus subtable values;
  - then `on`: images numeric criteria.
- **Speedup:** prep 9.0 → about 4 s CPU (c00, body.tex `s:prep`) and about 31 → about 10 s (calim07). Prep throughput per GPU 2.25 → 1.0 s/file (c00) and 7.7 → 2.5 s/file (calim07). This is the precondition for step 3, because prep is the next bottleneck once the GPU part is about 1.5× faster (body.tex).

### Step 3: peel through shared memory (`v3.peel_io: shm`)

- `v3/slots.py`: `SlotRing`, `write_slot(snap)`, `view(slot)`. Prep processes write their result straight into a free slot by name. The ring bounds the lookahead, replacing `GSI_PREP_LOOKAHEAD`.
- `workers/zest_daemon.jl`: `read_slot` / `write_slot!` and the `SHM` command (3.2). `stages/peel.py`: `ZestDaemon.zest_slot(path)` with a reply deadline (body.tex P1).
- After the peel, the parent writes the kept/avg MS from the slot with `write_ms_like` in the archive thread. The imager and the hour job stay on MSes in this step.
- **Test:** the peeled visibilities are **bitwise** identical to the v2 kept MS from the same prep (`v3_vis_compare`), and the images are within the floor.
- **Speedup:**
  - zest 3.10 → about 1.75 s (peel 1.63 + about 0.1 handoff), so c00 23 MHz goes 3.32 → about 2.0 s/file and the batch 1183 → about 710 s;
  - calim07: zest 9.6 → about 2.1 s, and the batch becomes prep-bound at about 2.5 s/file: 3745 → about 900 s (about 4×);
  - 55 MHz: 4.45 → about 3.5 s/file;
  - 82 MHz: about 5 % only (GPU-bound: 6.75 kernel-s), until the report's kernel items 3 and 4 land.

### Step 4: snapshot imaging from the slot (`v3.snapshot_io: shm`)

- New `workers/gsi_v3_engine.py`, a superset of `ttcalx_snapshot_daemon.py`. It has the ops `image_slot` and `close` and keeps `reuse_svd` and the pool limit. It runs `gpu_select` on the slot plus the in-memory LST rotation (3.3).
- `v3/engine_client.py`: a client like `gpu_image.SnapshotDaemon`, with ids, deadlines and restart on CUDA errors.
- `worker.py`: a `run_batch_v3` path (new file `v3/worker_v3.py`; `Worker.run_batch` dispatches to it when `peel_io == shm`). It drops the chgcentre copy. Kept MSes are still written when `write_kept_ms`.
- The engine runs with `hour_image.imager_env(cfg)` + `imaging.gpu_env`, so casacore measures and IERS data match the hour job.
- **Test:** snapshot I and V against v2. In-memory rotation against chgcentre is expected below 0.05 σ (numeric criteria). The hour products are still from MSes, so they are bitwise against step 3.
- **Speedup:** removes chgcentre (0.58 / 4.33 s CPU and 110 MB of writes per file) and the imager's MS read. Image wall 1.61 → about 0.7 s (c00) and 7.3 → about 1 s (calim07). The freed CPU goes to prep on loaded nodes: calim07 about 2.5 → about 2.2 s/file; c00 about 2.0 → about 1.9.

### Step 5: one archive file per band-hour, hour job reads it (`v3.archive: iv`, `hour_source: archive`)

- `v3/hourstore.py`: `RecordWriter`, `RecordReader`, GSIV v1 (section 4). The engine store thread appends records.
- `ttcalx_imager/gsiv.py` (TTCalX repo, python): `load_archive(path, ra, dec) -> list[LazySnap]`, plus `hour.image_hour_snaps` (3.4) and the CLI `ttcalx-image hour --archive <file>[,<file>]`.
- `hour_lst.run_band` builds that command when `hour_source == archive`; the archive lands under `QA/`.
- **Test:** run with `write_kept_ms: true` and the archive. The hour job on the archive against the hour job on the kept MSes of the **same** run must give bitwise-identical sample arrays (unit test: `load_samples(ms)` vs `load_archive` for every snapshot), and products within the floor.
- **As implemented (exp/gsi-v3-store, exp/imager-v3-store):**
  - one file per worker batch, `<archive_root>/<band>MHz/<date>/<run_label>/<LST>h/<band>MHz-<date>-<LST>h-peeled-bNNN.gsiv`, so two batches of one band-hour never append to the same file. The hour job reads every `*.gsiv` of the band-hour.
  - The worker hook (`Worker._archive_record`, before the FITS move) sends each peeled MS to `python -m gpu_subband_imaging.v3.hourstore serve` running in the imager venv (`v3.archive_python`, default `imaging.gpu_python` or the python beside `hour_imaging.ttcalx_image`). Records, their R and the hour job's own MS reading therefore use the same msio, casacore and numpy. A stamp that is already in the file is skipped.
  - QA rejections go to the imager by stem (`--exclude-stems`), not by index.
  - `hour_source: archive` needs `archive: iv`, and `hour_source: kept_ms` needs `write_kept_ms`. `archive: full` is rejected (not implemented).
  - With `--archive`, `TTCALX_CHUNK_CACHE` defaults to `store`. A `LazySnap` expands u, v, w, nw on the GPU in `HourGrid` in every mode.
- **Speedup:**
  - keep_ms 2.6 / 7.3 s of Lustre writes per file → about 0.05 s append;
  - hour load + rotation 46 → about 8 s (c00) and 171 → about 20 s (calim07);
  - **inodes 36.5 k → 2 per band-hour**;
  - Lustre writes 37 → 12.4 GB per band-hour.

### Step 6: hour stage in the same process (`v3.hour_source: memory`)

- **`v3/worker_v3.py`, end of batch:** close the zest daemon, QA from `self._qa`, engine `hour` op, then `place_products`, the movie thread and provenance.
- **`hour_lst.py`:** split `run_band` into `qa_exclusions()`, `run_imager_*()` and `finish()`, which v2 and v3 share. v2 behaviour is unchanged.
- **`orchestrator.py`:**
  - in `plan()`, do not seed hour-job rows for band-hours with `hour_source == memory`;
  - make `_maybe_wideband` count a DONE batch as the hour being done;
  - in `worklist.build`, use one batch per (band, LST hour) when `hour_source == memory`.
- **Test:** products bitwise against step 5 (same records), within the floor against v2.
- **Speedup:**
  - hour stage: no load (46 / 171 s), no separate job start (CUDA/CuPy/FFT-plan init plus dispatch, about 15–20 s, to be measured), the movie overlapped (14 / 41 s), no wait for a free slot;
  - hour stage 491 → about 430 s (c00) and 484 → about 270 s (calim07);
  - **band-hour totals: c00 23 MHz about 1674 → about 1140 s (about 1.5×); calim07 about 4229 → about 1060–1170 s (about 3.6–4×).**

### Step 7: memory policy, spill, checkpoint and restart (`v3.store`, `v3.checkpoint`)

- Section 5 policy in `hourstore.HourStore` (ram or memmap over the same file), the reservation file, the mid-run guard, and the resume logic (section 4) in `worker_v3`.
- Warm-gain persistence in `zest_daemon.jl`.
- Orchestrator node affinity when `checkpoint: local`.
- **Tests:**
  - `store: spill` against `store: ram`: bitwise-identical products;
  - kill -9 at snapshots 150 and 300, then rerun: products bitwise against an uninterrupted run;
  - torn-tail unit test (truncate mid-record, then the index crc check);
  - calim07 with both GPUs: no OOM, `V3_MEM_SPILL` logged when forced.
- **Speedup:** 0. Spill costs at most about 10 s per band-hour.

### Step 8 (optional): peel in-process through juliacall (`v3.peel_backend: juliacall`)

- The engine calls `TTCalX.zest_sky_rfi!` through juliacall.
- GPU handoff: `unsafe_wrap(CuArray, CuPtr{ComplexF32}(ptr), dims)` on CuPy's device pointer. Both use the device's primary context. Synchronise the CuPy and CUDA.jl streams at the handoff.
- Needs a TTCalX change: move `pycall_ms_bridge.jl` / `ms_io.jl` into a `PyCall` package extension. PyCall cannot coexist with juliacall's libpython.
- Also needs `PYTHON_JULIACALL_HANDLE_SIGNALS=yes` and the existing Julia 1.10.4 and depot (`PYTHON_JULIAPKG_OFFLINE=yes`).
- **Test:** bitwise against step 7.
- **Gain:** −0.1 to −0.2 s/snapshot, and one GPU memory manager instead of two competing pools (body.tex P2, P3). Medium risk; do it last.

### Step 9 (optional): incremental hour setup

- Density grids per interval accumulated during the batch.
- Gain about 5–10 s per band-hour. Low priority.

## 8. Expected effect per night (estimate)

- **v3 alone (steps 1–7):**
  - batch slot-hours (77 slot-h in body.tex) drop about 35–40 % for bands ≤ 41 MHz, about 20 % at 46–59 MHz and about 5 % at 64–82 MHz, where the GPU kernels bind;
  - hour slot-hours drop 37 → about 32 (load 50–84 s × 165 band-hours plus startup and movie);
  - about **114 → about 90 slot-h**, with far larger gains on memory-pressured nodes such as calim07;
  - plus the end of the 6.5 TB/night of kept-MS writes, the same again in reads, and about 36 k inodes per band-hour.
- **v3 plus the report's kernel items 3, 4 and 7** (imager w-term −1.5 s at 82 MHz, mixed-precision peel 3.2 → 0.5–1 s, batched hour gridding): with the disk path gone, faster kernels shorten the critical path directly. Expect about 50–60 slot-h, i.e. **about 3–4 h of wall time on 16 GPUs**. That is consistent with the body.tex v3 target of 2–3 h once the follow-mode scheduling is in.

## 9. Risks

- **mstransform equivalence of the numpy averaging** (step 2): validate bin by bin in shadow mode, including all-flagged bins and the WEIGHT/SIGMA definitions (`fastcal.py:105` comment).
- **No per-snapshot MSes:** orca tools that need visibilities (hot baselines, AOFlagger post-concat) need `archive: full` + `export-ms`, or in-engine statistics.
- **Batch granularity:** one batch is one band-hour (about 20–45 min). A late failure costs more. The local checkpoint and replay limit this.
- **Two processes per node** must coordinate RAM through the reservation files. The `auto` policy assumes MemAvailable is honest; with swap full on calim07, keep `ram_reserve_gb` ≥ 16.

### Critical files for implementation
- `/private/tmp/claude-501/-Users-nikita-caltech-courses-distributed-pipeline/38f3812d-2c74-4a36-90ed-8f2853117c8f/scratchpad/e2e/code_new/gpu-subband-imaging/src/gpu_subband_imaging/worker.py`: `run_batch` / `_preprocess` / `_image_archive`, plus the new `v3/worker_v3.py`, `v3/fastprep.py`, `v3/slots.py`, `v3/hourstore.py`, `v3/engine_client.py`.
- `/private/tmp/claude-501/-Users-nikita-caltech-courses-distributed-pipeline/38f3812d-2c74-4a36-90ed-8f2853117c8f/scratchpad/e2e/code_new/gpu-subband-imaging/workers/zest_daemon.jl`: `SHM` command, `read_slot` / `write_slot!`, warm-gain persistence. Reference: `TTCalX-sponge/src/gpu/ms_io.jl` (`read_ms_fast` / `write_ms_fast!`).
- `/private/tmp/claude-501/-Users-nikita-caltech-courses-distributed-pipeline/38f3812d-2c74-4a36-90ed-8f2853117c8f/scratchpad/e2e/code_new/gpu-subband-imaging/workers/ttcalx_snapshot_daemon.py`: becomes `workers/gsi_v3_engine.py`.
- `~/caltech_courses/ttcal/julia_ttcal/TTCalX-imgspeed/python/ttcalx_imager/src/ttcalx_imager/hour.py`, `msio.py`, `snapshot.py`: `image_hour_snaps`, `LazySnap` / `_n`, `TTCALX_CHUNK_CACHE=store`, the `msdata_from_arrays` refactor, the new `gsiv.py`.
- `/private/tmp/claude-501/-Users-nikita-caltech-courses-distributed-pipeline/38f3812d-2c74-4a36-90ed-8f2853117c8f/scratchpad/e2e/code_new/gpu-subband-imaging/src/gpu_subband_imaging/config.py`, `hour_lst.py`, `orchestrator.py`, `stages/uvw.py`, `stages/fastcal.py`, `stages/stage_ms.py`: `V3Params`, the `run_band` split, hour-row seeding and affinity, the array versions.
- `/lustre/nkosogor/peel_exp/e2e/e2e_compare.py` (on calim07): the base for `scripts/v3_compare.py`.
