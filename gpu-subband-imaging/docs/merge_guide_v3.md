# Merging gsi v3 (quick guide)

## What to merge, in order

| # | repo | PR | contains |
|---|---|---|---|
| 1 | TTCalX | `feature/ttcalx-v3` -> `main` | `feature/gpu-clean-imager` (production imager, 748884a) + `feature/sponge-restore-fast-zest` (peeler 0.3.0) + v3 imager work, in one commit. Supersedes both branches. |
| 2 | distributed-pipeline | `feature/gsi-v3` -> `main` | `feature/gsi-lst-nightly` (= v2, 13405ce, 30 commits not yet in `main`) + v3. Merging it brings v2 into `main` too. If you want v2 in `main` on its own first, merge `feature/gsi-lst-nightly` before it. |

Deploy TTCalX and gsi together. **Never check out the old `exp/ttcalx-v3` alone in the live TTCalX tree**: it has no imager, and the production imager venv is an editable install of that tree.

Not part of v3 (separate, older, review on their own or leave): `prod-provenance`, `feature/gsi-streaming` (`run_backlog.py`), `exp/gpu-peeling-experiments` (test scripts).

## Why a line-by-line review isn't needed

- **Defaults = production.** With no `v3:` block, no new `peel:` option and no new env variable, gsi runs as 13405ce and TTCalX as 748884a. Every remaining default-path difference is listed in `default_path_vs_13405ce.md`.
- **Checked by measurement:**
  - default peel output is bitwise equal to 748884a;
  - the 82 MHz hour job uses no more GPU memory (8.8 vs 9.3 GB) and its images are within 748884a's own run-to-run floor;
  - a 55 MHz default-path smoke run equals the production night (QA rows, provenance and log lines equal);
  - the run JSON, `ZEST_*` env and provenance equal 13405ce's.
- **Tests:** gsi 329 + 7 Julia; imager 115 (GPU); TTCalX 1306 (GPU).
- **v3 itself:** all subbands of 2026-10-31 LST 05h are at the floor vs production, with equal fluxes, QA and Jupiter, about 2.5x faster per band-hour.
- **Reviews:** five during development, two independent reviews and a re-check, all addressed.

## Where to look (10 minutes)

1. The README "Versions" and "v3" sections: the switches and their defaults.
2. `docs/default_path_vs_13405ce.md`: the table of what still differs on the default path and why it is harmless.
3. Always-on TTCalX imager changes: compact w-term factors and new Hogbom kernels. These were validated within the GPU floor; the snapshot-path GPU memory was not measured separately.
4. New v3 safety defaults: `v3.max_given_up_fraction` 0.1, `v3.stale_checkpoint_hours` 48.

## Known and left for later (v3 / opt-in only)

- `PeelWorkspace` never shrinks (about 2–4 GB of GPU memory held for the daemon's lifetime).
- Warm-start age check.
- `_is_gpu_oom` matches error text.
- `use_fast_settings` changes `os.environ` for the rest of the process (only in-process test scripts are affected).
- GSIV repair truncates at the first bad record.
- The archive is never deleted.
- `engine=sky_rfi` ignores the `ZEST_REG_*` phase prior.
- Existing production behaviour, not new: the GPU CLEAN can flip a faint component between runs, giving up to 0.6σ single-pixel differences in 10-min images.

## Roll back

```bash
git checkout gsi-v2        # distributed-pipeline (13405ce)
git checkout gsi-v2        # TTCalX (748884a), the tree the imager venv points to
```

`gsi-v1` in both repos is the deployment that ran until 2026-10-25.
