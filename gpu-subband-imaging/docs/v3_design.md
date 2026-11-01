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

R is not bit-reproducible in this sense. casacore measures caching makes `snapshot_rotation` depend on the order of the calls in a process, by |ΔR| up to about 1.8e-10 (`scripts/v3_store/r_history.py`), a phase of about 1e-7 rad. A record's R equals a sequential `load_samples` in the writer's order (`hourstore build` sorts by stem), but not, bit for bit, a process-pool load. v2's own hour job (`load_all`, nproc 8) has the same property between any two runs, so this sits inside the v2 floor.

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
- **Done (exp/gsi-v3-prep):** `stage_ms.stream_untar`. Checks: members only under `<stamp>.ms/`, file sizes, the end-of-archive marker (tarfile stops silently at a cut), all bytes read. Use a 64 KiB stream buffer with 1 MiB copies; tarfile re-slices its buffer per read, so 4 MiB buffers cost 1.4 s CPU per tar. Measured on calim10 (2026-11-01, 72 tars): 0.22–0.29 s CPU against 0.35–0.46 s for copy + extract. The staged raw MSes are byte-identical.

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
- **Done (exp/gsi-v3-prep), with corrections to the plan above:**
  - mstransform weights DATA by nothing. Its output weights come from **SIGMA** (1/SIGMA² per channel); the input WEIGHT column is ignored. The exact rules are in the `v3/fastprep.py` docstring.
  - mstransform rewrites every subtable *description* (column comments, PROCESSOR without SETTINGS, ...) but copies the *values*. So `write_ms_like` copies the template file by file and fills the values from the raw MS. It does not copy raw subtables.
  - **Result on 72 real snapshots (23/55/82 MHz, 10-31):** bitwise identical to v2 in DATA, FLAG, FLAG_ROW, WEIGHT, SIGMA and UVW. Table structure and all subtable values are identical. Median CPU per file, v2 → v3 including `write_ms_like`, with 4 parallel processes on calim10: 23 MHz 5.06 → 2.43 s, 55 MHz 4.79 → 2.28 s, 82 MHz 7.15 → 4.55 s. Results: `/lustre/nkosogor/peel_exp/v3/prep/results`.

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
- **As implemented (exp/gsi-v3-engine, exp/imager-v3-engine):**
  - `workers/gsi_v3_engine.py` (imager venv, pipeline src on PYTHONPATH): ops `store_open`, `image_slot`, `spill`, `hour`, `info`, `close`; replies `GSI_V3 <json>` with the request id, only after nothing reads the slot any more (FITS written and record stored), so the worker may then reuse it. GPU work on the main thread, records in a store thread, FITS in a writer thread; every casacore measures call stays on the main thread.
  - **In-memory chgcentre** (`rotate.chgcentre_uvw`, `rotate.parse_phase_centre_chgcentre`, `snapshot.rephase`): UVW = A[a1] − A[a2] from casacore measures at TIME (what chgcentre writes), data · exp(2πi (w − w′) f / c) in complex128 stored as complex64, PHASE_DIR from chgcentre's own parse (aocommon long-double RA/Dec, then casacore direction cosines). Checked against /opt/bin/chgcentre (`scripts/v3_engine/chgcentre_check.py`, 9 real MSes, 23/55/82 MHz): UVW within 2.0e-7 m, DATA rel. rms 0.8–4.3e-8 (max 6.3e-6 of the rms: float32 rounding), FLAG/WEIGHT/other columns untouched, PHASE_DIR bit-identical for all 24 LST-hour centres (astropy's parse differs by 1 ulp in 20 of them). The R of the hour record is computed separately with `parse_phase_centre`, exactly as `load_samples` does.
  - The snapshot itself goes through `snapshot.gpu_select` and `SnapshotImager.compute / write` (= `_compute` / `_finish`), as the v2 daemon's raw reader path.
  - `v3/engine_client.py`: thread-safe, ids, per-request deadline (`v3.engine_timeout_s`), kill + restart on a timeout, an exit or a "fatal" (CUDA context) reply, stale replies of a killed engine never taken, `on_restart` re-opens the store (the engine reloads the checkpoint file), at most 3 restarts. A snapshot in flight during a restart is retried once (the slot is untouched: the engine maps it read-only).
  - `v3/worker_v3.py` (`Worker.run_batch` dispatches to it when `snapshot_io=shm`): see step 6.
  - **Result (lwacalim00 GPU 1, 2026-11-01, 24 snapshots per band, the same peeled MSes):** slot + engine (S) against v2 copy + chgcentre + snapshot daemon (A), and A against itself (B, the floor). Max over the snapshots, diff in units of the image sigma: I diff rms 5.0e-7 / 1.7e-5 / 7.1e-6 (23 / 55 / 82 MHz; floor 3.5e-7 / 7.7e-5 / 2.7e-7), I diff max 1.5e-5 / 1.1e-3 / 4.8e-4; V diff rms ≤ 2.2e-6, max ≤ 1.0e-5. Noise ratio within 2.5e-6, check-source peaks (SNR > 10) S/A within 1 ± 5.4e-7, headers equal except BMAJ/BMIN/BPA at ~1e-8 relative (the beam fit). Engine image wall 0.34 / 1.15 / 2.42 s against v2 0.43 + 0.41 (copy + chgcentre) / 1.23 + 0.41 / 2.49 + 0.41 s. Results: /lustre/nkosogor/peel_exp/v3/engine/results/a_*.json.

### Step 5: one archive file per band-hour, hour job reads it (`v3.archive: iv`, `hour_source: archive`)

- `v3/hourstore.py`: `RecordWriter`, `RecordReader`, GSIV v1 (section 4). The engine store thread appends records.
- `ttcalx_imager/gsiv.py` (TTCalX repo, python): `load_archive(path, ra, dec) -> list[LazySnap]`, plus `hour.image_hour_snaps` (3.4) and the CLI `ttcalx-image hour --archive <file>[,<file>]`.
- `hour_lst.run_band` builds that command when `hour_source == archive`; the archive lands under `QA/`.
- **Test:** run with `write_kept_ms: true` and the archive. The hour job on the archive against the hour job on the kept MSes of the **same** run must give bitwise-identical sample arrays (unit test: `load_samples(ms)` vs `load_archive` for every snapshot), and products within the floor.
- **As implemented (exp/gsi-v3-store, exp/imager-v3-store):**
  - one file per worker batch, `<archive_root>/<band>MHz/<date>/<run_label>/<LST>h/<band>MHz-<date>-<LST>h-peeled-bNNN.gsiv`, so two batches of one band-hour never append to the same file. The hour job reads every `*.gsiv` of the band-hour.
  - The worker hook (`Worker._archive_record`, before the FITS move) sends each peeled MS to `python -m gpu_subband_imaging.v3.hourstore serve` running in the imager venv (`v3.archive_python`, default `imaging.gpu_python` or the python beside `hour_imaging.ttcalx_image`). Records, their R and the hour job's own MS reading therefore use the same msio, casacore and numpy. A stamp that is already in the file is skipped.
  - QA rejections go to the imager by stem (`--exclude-stems`), not by index.
  - **Order in `_image_archive`:** FITS fpacked and moved → the stem appended to `<archive>.expected.jsonl` → keep_ms → the record, made from the kept MS. The archive's snapshot set is therefore v2's kept set. At batch start the writer's imager is probed (`hourstore.probe_imager`); without `ttcalx_imager.gsiv` the batch fails at once.
  - **Writer:** requests carry ids. On a timeout or a dead child the client kills its `serve`, the worker drops it and the next snapshot starts a new one, which resumes the file. With `hour_source=kept_ms` the failure is logged loudly once per batch.
  - **Reruns:** a reprocessed stem is appended again; the newest record wins by (written_ns, seq), then path, also across bNNN files. A resume refuses another band, date, group, run_label, calibration table, phase centre or channelisation. Each file has an O_EXCL lock file with pid and host, touched on every append and taken over when stale.
  - **Strict loading:** torn bytes, dropped index lines, unindexed records, a bad crc or a record of another band raise `GSIVError` (array tables are bounds-checked whether or not the crc is verified). `load_archive(band=...)` checks the header band.
  - **The hour job never images a reduced set.** It verifies the archive structure and crc, the band and the phase centre, and that the newest-per-stem set equals the expected set (worker lists ∪ CLEAN snapshot FITS) and the kept MSes. Otherwise it falls back to the kept MSes when they cover the expected set, and fails loudly when they don't. With `hour_source: archive`, kept MSes are deleted only when the imager loaded exactly the stems it was given, and then only those stems. At the defaults (`kept_ms`), deletion is unchanged from 13405ce: on success the band-hour's kept directories are removed whole. A test compares old and new `run_band` on the same fixtures. No-data and failed jobs delete nothing. See `scripts/v3_store/corruption_table.py`.
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
- **As implemented:**
  - **Batch** (`v3/worker_v3.py`): feeders stream the tar into tmpfs and run `fastprep.prep_slot_task` in the spawn pool, which writes the SnapArrays straight into a ring slot (no `write_ms_like`). The ring has `lookahead + 2` slots: a feeder gets a slot before it starts (in order, the next one blocking only when the peel needs it), so at most `lookahead` snapshots are prepped ahead and two are with the engine; `GSI_PREP_LOOKAHEAD` is not used. The daemon peels the slot in place (`zest_slot`, PEELED check), then a post thread sends `image_slot` and, with `write_kept_ms` (and a keep root), writes the kept MS from the slot in a pool process (`fastprep.kept_from_slot_task`, the raw MS is kept in tmpfs until then); the slot is released after both; the FITS go through `Worker._post_image` (frame, pilot QA row, fpack, move) as v2. With records, the FITS are moved only after the record is stored: a snapshot is done only with both.
  - **Hour stage** (`hour_source=memory`): daemon closed, prep pool stopped, the pilot rows written to `.snapshot_stats` and merged with those of earlier runs, `hour_lst.qa_step`, the movie (`hour_lst.movie_step`) in a thread, the engine's `hour` op with the v2 hour job's argv (`ttcalx_imager.cli.run_hour_snaps`, which shares `hour_setup` with the CLI, so the products, phase centre and options are those of `ttcalx-image hour`; `TTCALX_CHUNK_CACHE=store`; `TTCALX_CACHE_RESERVE=auto`, so the device chunk cache leaves the multiscale CLEAN its workspace, and the imager images an interval again without the device cache after a GPU OOM; the fallback is in provenance as `hour_cache_fallback`), the expected stems = the band-hour's CLEAN snapshot FITS (the engine refuses when a record is missing), QA rejections by stem; `place_products`, `hour_lst.write_provenance` with a `v3` block (modes, store plan and path, n_records, engine restarts, worker / engine peak RSS, MemAvailable at start and end), kept MSes deleted per imaged stem with `delete_kept_ms`. A failed hour stage fails the batch (provenance status failed; the checkpoint stays for the retry).
  - **Orchestrator:** `hour_source=memory` → `worklist.build` makes one batch per (band, LST hour), `plan()` seeds no hour-job rows, `_maybe_wideband` counts the band-hour's batch DONE as its hour.
  - `hour_lst.run_band` is split into `qa_step`, `movie_step`, `write_provenance` (shared); v2 behaviour unchanged (the suite's hour-job tests pass unchanged).
  - **Result (23 MHz, LST 05h, 10-31, all 358 inputs, lwacalim00 GPU 1, real orchestrator):** 356 imaged, 2 skipped (no data), 0 failed, QA identical to the e2e run (36 rejected). Band-hour dispatch to done 934 s against e2e "new" 1308 s (batch 981 + separate hour job 317 s): snapshots 653 s (peel-bound, zest 1.81 s/file), hour stage 250 s (load 0.1 s against 33.7 s). Hour products, memory replicas M1, M2 (M2 = the same records reloaded from the checkpoint) against the v2 hour job on the kept MSes of the same run, twice (A, B): deep A:M2 diff rms 4.2e-7 sigma (A:B 1.9e-4), 10 min A:M2 7.6e-6 sigma (A:B 5.0e-5); one M1 10-min image (interval 4) differs from all three others by rms 0.024 sigma / max 1.15 sigma, a single-run CLEAN divergence (A:M2, B:M2 are at the floor); check-source peaks within 1 ± 5e-4 everywhere, noise within 2.2e-4. Worker peak RSS 0.56 GB, engine 9.6 GB (the 7.2 GB RAM store included), Julia 2.3 GB, prep processes 2.7 GB. GPU (nvidia-smi, my processes): about 5.0 GB median / 5.2 GB max while the daemon and the engine both run (snapshots, 23 MHz); 12.2 GB in the hour stage, which runs alone (the daemon is closed first). Daemon + engine at 82 MHz has not been measured yet (next GPU run). Results: /lustre/nkosogor/peel_exp/v3/engine/results/b_hour_compare.json.

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
- **After review (2026-11-01):**
  - **Engine deaths:** a request that dies with others in flight is retried *alone* (an exclusive gate: nothing else in the engine), so a healthy sibling is not blamed; one that dies again alone is the culprit (`V3_ENGINE_CULPRIT`) and fails. `v3.engine_max_restarts` (5) exhausted, or a restart whose store re-open fails -> `EngineFailed`, `V3_ABORT`: the batch stops at once (no more prep / peel). In memory mode an engine death without a record file aborts too (`checkpoint: none` is refused with `hour_source: memory`; no spill dir with room is logged `V3_NO_CHECKPOINT`).
  - **Never a reduced hour:** in memory mode any failed snapshot (prep error, peel, engine) fails the batch (`V3_INCOMPLETE`) before the hour stage; the orchestrator's retry redoes only the missing snapshots (FITS + record = done). Snapshots skipped for no data are listed in `QA/.v3_skipped.jsonl` and not prepped again; a band-hour whose hour is done (provenance) is not touched.
  - **Ledger:** `run_key` gets `_v3mem` with `hour_source: memory` (one batch per band-hour, no hour-job rows), so a v2 ledger of the same run is never reused.
  - **Budget:** with `hour_source: memory`, `batch_timeout_seconds` must be at least `hour_imaging.timeout_seconds` + 1800 s + `v3.budget_margin_s` (300 s); the worker stops itself (`V3_BUDGET`: daemon, engine and prep processes killed, `os._exit(124)`) `budget_margin_s` before the orchestrator's timeout, and the hour deadline is what is left of that budget. A retry on another node therefore never overlaps a still-running orphan; it finds the FITS on Lustre (skipped), has no local checkpoint (replays the records: prep + peel), and rewrites the hour products and provenance (`place_products` moves over the files of an earlier attempt). Pilot QA rows are appended to `.snapshot_stats` per snapshot, so a killed attempt keeps them.
  - **Replays re-peel:** a replayed record comes from a new peel of the snapshot, not from the peel that made its FITS (the peel is not bit-reproducible on the GPU); replays happen only for snapshots whose record is not in the checkpoint, so there is nothing to replay "from the checkpoint".
  - **Memory plan:** counts every input of the band-hour (records already in the checkpoint at its own bytes per record, replays, todo) plus a fixed 15 GB per process (prep, Julia, engine); the checkpoint's own dir is kept. `/dev/shm` is checked for the ring and the tmpfs raw MSes (`V3_SHM_TIGHT` -> untar to work_root). The guard checks every `mem_check_every` (2) snapshots and escalates each time: spill, then lookahead 2 and untar to work_root, then logs.
  - `write_kept_ms` now defaults to False with `hour_source: memory` (True otherwise).
- **Follow-up review (2026-11-01):**
  - **Inputs that fail every attempt:** every failed input of an attempt (prep error, peel failure, engine culprit / archive failure) is appended to `QA/.v3_failed.jsonl` (stamp, reason, permanent, batch, host, time). An input that failed in `v3.max_input_failures` attempts (0 = `max_retries` + 1), or whose tar has vanished (permanent at once), is *given up*: it is not prepped again, and the hour is made from the rest (`V3_HOUR_REDUCED`; provenance `hour.failed_inputs` with attempts / reason, `hour.hour_reduced`). Before that, a failed input still fails the attempt (`V3_INCOMPLETE`). The hour refuses (`V3_UNACCOUNTED`) when an input is neither imaged (CLEAN FITS), skipped for no data, nor given up.
  - **tmpfs cleanup:** the sweep of crashed workers' untar dirs and every cleanup use the configured `v3.untar_to` root (`Worker._untar_root(configured=True)`), also after the batch turned tmpfs untar off (`V3_SHM_TIGHT`, the memory guard).
  - **Profiling:** `v3.profile_no_image` needs `GSI_V3_ALLOW_PROFILE=1` (forwarded by the orchestrator) and always fails the batch after the snapshots (`V3_PROFILE`): no FITS, no provenance, never "done".
  - **Hour retry:** also after an engine death (a CUDA fatal error); not for the deterministic "expected snapshot(s) have no record" refusal; not when less than max(300 s, half the first attempt) of the batch budget is left; with a clear error when the engine cannot be restarted (restarts exhausted, or the budget kill started).
  - **Budget:** the timer is cancelled on every exit path; `v3.budget_margin_s` must be smaller than `batch_timeout_seconds`; once the budget kill starts the engine client is marked failed (no restart).
  - **Reservations:** a batch reserves only its RAM records (0 when spilled); the 15 GB fixed per-process term is in the other workers' MemAvailable already and is no longer counted twice.
  - **Old checkpoints:** a checkpoint named with the run_key before its `_v3mem` suffix is moved to the new name at the next batch start and reused; a band-hour whose hour is done deletes its local checkpoints (both names, `v3.delete_checkpoint`).
  - **Pilot rows:** carry their write time `t`; `hour_lst.read_stats` keeps the newest row of a snapshot by time (file modification time for rows without `t`), not by file name.
  - **Defaults of the other `snapshot_io=shm` modes (L5): kept, documented.** `v3.mem_check_every` is 2 (was 10), `v3.engine_max_restarts` 5 (was 3, now with the culprit rule: a culprit costs two restarts), and the memory guard acts without records too (lookahead 2, untar to work_root; the spill step needs records). These apply to every `snapshot_io=shm` batch, `hour_source` kept_ms / archive included: the same engine and slot ring, so the same limits; the checks are cheap (one /proc/meminfo read) and only make the batch more careful under memory pressure. v2 (`snapshot_io=ms`) is unaffected.
- **GPU re-validation after the review fixes (lwacalim00 GPU 1, 2026-11-01, memory mode, write_kept_ms false, defaults otherwise):**
  - 23 MHz, 358 inputs: band-hour (dispatch to done) 791 s against the e2e "new" 1308 s (snapshots 520 s, hour 245 s). Hour products against the two v2 hour jobs on the kept MSes of the first run: deep diff rms 2.1e-4 sigma, 10 min 5.0e-5 (v2 A:B 1.9e-4 / 5.0e-5); snapshots against e2e "new" (every 10th): I rms <= 0.0017 sigma, V <= 0.0028, peaks within 2e-4.
  - 82 MHz, 358 inputs: band-hour 1204 s against e2e "new" 1734 s (snapshots 704 s against batch 1217 s; hour 470 s against 473 + the separate job's ~40 s). Hour products of a kept-MS run (two memory replicas, the second from the checkpoint) against two v2 hour jobs on its kept MSes: deep rms <= 5.3e-4 sigma, 10 min <= 5.6e-3 (v2 A:B 5.1e-4 / 5.6e-3; single-pixel maxima up to 0.35 sigma occur between the v2 replicas too); peaks within 1e-4; snapshots against e2e "new": I rms <= 0.009 sigma, V <= 0.013, peaks within 0.16 %.
  - GPU memory (nvidia-smi, my processes): daemon + engine median 5.2 GB (23 MHz) / 5.4 GB (82 MHz); the hour stage alone peaks at 12.3 GB / 15.95 GB (of 16 GB: as v2's hour job, the hour stage has no headroom at 82 MHz). Peak RSS: engine 9.6 / 6.9 GB (the RAM store included), worker 0.6 GB, Julia 2.6 / 2.3 GB, prep processes 3.8 / 3.2 GB together.
  - **Fixed on the way (82 MHz):** the first hour attempt failed in cuFINUFFT setpts ("Error setting non-uniform points") while the snapshot imager's cuFINUFFT plans and w-term caches were still on the GPU. The engine now clears every imager GPU cache before the hour (`gridder.clear_caches()`, gc), and the worker retries a failed hour once in a fresh engine (`V3_HOUR_RETRY`, records reloaded from the checkpoint). The 82 MHz timing run then made its hour in the long-lived engine at the first attempt.
- **Peel-loop profiling (23 MHz, lwacalim00 GPU 1, 2026-11-01).** The v3 zest phase was 1.81 s/snapshot (1.45 s without kept MSes) against v2's read 0.94 + peel 0.75 + write 0.51:
  - The shm hand-off works: daemon read 0.13 s, write 0.11 s (0.31 / 0.33 s while the kept-MS writer competes for the CPU in the prep pool).
  - The PEEL_SOLVE work is identical to v2 (18132 solver iterations, 2536 accepted), but the peel itself is slower: median 1.09 s against 0.60 s.
  - **Not GPU contention with the engine:** with imaging serialised behind the peel (`v3.image_overlap: false`) the peel median is 0.92 s; with no imaging at all (`v3.profile_no_image`) 1.01 s.
  - **Not the prep pool:** isolated daemon runs (`v3_shm_peel_equiv.py peel`, 24 snapshots, no engine) give MS path 0.60 s, SHM path 0.58 s, and SHM with 4 concurrent prep processes 0.55 s. Node load during the batch is about 7 of 48 CPUs.
  - **The cause is the daemon's `JULIA_CUDA_SOFT_MEMORY_LIMIT=4GiB`** (set by the worker since the GPU imager shares the card): isolated SHM peel 0.58 → 0.90 s with it; an idle 8 GB CuPy allocation beside it changes nothing (0.57 s). With the MS path, the ~1.4 s of casacore I/O between peels gave Julia's GC idle time; with the shm hand-off the pool reaches the limit and CUDA.jl collects inside the solve.
  - **Switches** (defaults = v2): `v3.zest_soft_memory_limit` ("" = 4GiB as v2, "none", or a value), `v3.zest_hard_memory_limit`, `v3.zest_gc_between` (young | full, `ZEST_GC_BETWEEN` in the daemon: GC after each reply), `v3.image_overlap`, `v3.profile_no_image`. Batch measurements (80 snapshots), median zest / peel-loop period per snapshot / GPU max with the daemon:
    - defaults: 1.35 s / 1.31 s / 8.2 GB
    - no limit: 1.03 s / 1.16 s / **15.0 GB**
    - 8 GiB soft: 1.40 s / — / 8.5 GB
    - 4 GiB + full GC between: 1.34 s / 1.37 s / 5.6 GB
    - 6 GiB + full GC between: 1.36 s / 1.38 s / 7.1 GB
    - hard 5 GiB, no soft: 1.34 s / 1.40 s / 6.5 GB
    - imaging serialised: 1.51 s / 1.83 s
  - **Conclusion:** the GC moves (into the solve, or after the reply) but is not avoided while the daemon's pool is bounded; only an unbounded pool removes it (−0.15 s/snapshot, −11 %), at the cost of the daemon growing to fill the GPU (15 GB with the engine at 23 MHz), which is not safe at 82 MHz. Defaults unchanged. The real fix is a TTCalX change: preallocated GPU workspaces in the sky / RFI peel, so a snapshot allocates (almost) nothing. Not done here.
- **As implemented (minimal):** `v3/memory.py`: `plan_store` (`store: ram | spill | auto`; auto = RAM if n_todo × record bytes × 1.15 ≤ MemAvailable − `ram_reserve_gb` − the other gsi v3 processes' reservations, `/dev/shm/gsi3_mem_<uid>_<pid>.json`; else memmap from a GSIV file in the first `spill_dirs` entry with need × 1.2 free, never Lustre), `MemoryGuard` (every `mem_check_every` snapshots; MemAvailable below the reserve → engine `spill`, lookahead cut to 2, `V3_MEM_SPILL`). `HourStore.spill` turns a RAM store into memmapped records (writing the file first if there is none). The checkpoint: with `checkpoint: local` the engine appends every record to `<spill dir>/<run_key>/<band>MHz-<date>-<LST>-bNNN.gsiv` (with `archive: iv` the archive file is the checkpoint); at batch start the engine repairs and loads it, snapshots with FITS and record are skipped, those with FITS but no record are replayed (prep + peel + record, no imaging; `V3_REPLAY`), and if the hour products are already done (provenance hour status ok, source memory) the batch has nothing to do. The checkpoint is deleted after the hour products (`delete_checkpoint`). Not done: the warm-gain persistence in the daemon and node affinity for retries (a retry elsewhere replays).

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
