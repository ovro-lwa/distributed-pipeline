# Peel configuration (sky_rfi engine): recommended settings and the ablation behind them

This note is for the `peel:` section of `pipeline.yaml` with `engine: sky_rfi`
(TTCalX `zest_sky_rfi!`, branch `feature/sponge-restore-fast-zest`).

- The measurements were made on CPU on 2026-11-01. They use 2026-10-31 data, LST 05h, at 23, 55 and 82 MHz.
- The Jupiter check uses the 2024-12-20 and 2024-12-27 burst snapshots.
- Results are in `/lustre/nkosogor/peel_exp/ablate/`: `SUMMARY.txt`, `results/`, and the scripts in `scripts/`.

## Recommended `peel:` section

```yaml
peel:
  # CPU-TTCal-matched zest, as production
  maxiter: 5
  tolerance: 0.0001
  minuvw: 5.0
  min_source_elevation_deg: 0.0
  require_convergence: false
  max_gain_amplitude: 100.0
  solver: rk4
  model: exact
  uvw: geometric
  flag_model: true
  resubtract: true
  fast_io: true
  rfi: true
  flag_zero: true
  residual_guard: true
  engine: sky_rfi
  peeliter: 2                  # same quality as 3, about 40% less peel time
  precision: mixed             # same results as double, 25-30% less peel time
  warm_start: false            # drifts over long snapshot chains (see below)
  # band-gain smoothing only where it helps (below 65 MHz)
  smooth_flux: 1000.0
  smooth_exempt_flux: 500.0    # sources >= 500 Jy (catalog, band centre) keep per-channel gains;
                               # helps Tau A when it is high, hurts it at low elevation (see the Jupiter check)
  smooth_below_mhz: 65.0       # subband centre frequency; 64 MHz band (centre ~66.4 MHz) and up: off
  # Jupiter restore below 41 MHz (one JSON per night, position from the ephemeris)
  restore: /path/to/jupiter_<date>.json
  restore_below_mhz: 41.0
  restore_offset_deg: 0.5
```

`peel_overrides` can do the same as the frequency gate, for example `{64: {smooth_flux: 0}, 69: ..., 82: ...}`.
The gate is simpler and is logged on every snapshot as
`PEEL_SMOOTH_GATE centre_mhz=... below_mhz=... smoothing=on|off`.

## The switches

All three are sent to the daemon only when set, so the defaults leave the daemon environment and its output unchanged.

| key | default | meaning |
|---|---|---|
| `smooth_below_mhz` | 0 (off) | apply `smooth_flux` and `rfi_smooth_flux` only when the subband centre (the mean channel frequency) is below this value |
| `smooth_exempt_flux` | 0 (off) | sky sources whose catalog Stokes I at the band centre is at least this value keep per-channel gains. This is TTCalX `smooth_exempt_flux`, where 0 maps to `Inf` |
| `rfi_smooth_flux` | unset (= `smooth_flux`) | smoothing flux for the RFI emitters (TTCalX `rfi_smooth_flux`) |

- Every value must be at least 0. A negative value raises `ConfigError`, and the daemon exits with `CONFIG_ERROR`.
- The `SKY_RFI` settings line shows all of them and now reaches the `gsi.peel` log.
- In the 7-source model, catalog flux at about 25 MHz is: Cas A and Cyg A above 10 kJy, Vir A about 7 kJy, Tau A about 3.2 kJy, Hya A and Her A about 600 Jy, Per B (3C123) 400 Jy.
- Exempt 500 therefore leaves only Per B and the RFI emitters band-tied.

Checks, on CPU through the real daemon (lwacalim10):
- 82 MHz with `smooth_below_mhz: 65` is bitwise equal to `smooth_flux: 0` (DATA and FLAG, 2 snapshots).
- The default settings give bitwise the same output as the previous daemon (82 MHz new config; 23 MHz production).
- 23 MHz with `smooth_exempt_flux: 500` reproduces the ablation run E8 to 3e-8 Jy.

## How the ablation measured

- **Estimator:** the visibility-space apparent flux, i.e. the natural-weighted dirty pixel. It is the flag-aware mean of (XX+YY)/2 · exp(−2πi(ul+vm+w(n−1))) at each source position.
  - MS UVW are J2000 values from fix_uvw, A[ANTENNA1] − A[ANTENNA2].
  - Stokes V is i(YX−XY)/2, as in the ttcalx imager.
  - Baseline cuts are |uv| > 10 λ and > 60 λ.
- **Reported quantities:** peeled / unpeeled at each position. For the peeled sources lower is better; for the check sources higher is better.
- **Blank-sky noise:** the rms of the dirty I and V at 120 random points above about 20° elevation and more than 3° from any listed source, relative to production.
- **Validation:** the peak is within 0.1° of every catalogue position, and the opposite phase sign gives about 0.

## Ablation, 2026-10-31 LST 05h

### Medians at 10 λ / 60 λ

- 23 MHz: 12 snapshots. 55 MHz: 6 snapshots.
- Tau A, Cas A and Per B are residuals. Per A (3C84) and 3C134 are retained check sources.
- blankV is the 10 λ blank-sky V rms relative to production.
- s is the peel wall time at 6 CPU threads.

| 23 MHz | Tau A | Cas A | Per B | Per A | 3C134 | blankV | s |
|---|---|---|---|---|---|---|---|
| production (= sky_rfi with production options) | −0.007 / 0.021 | −0.009 | 0.31 | 0.41 | 0.37 / 0.55 | 1.00 | 17.4 |
| + smooth_flux 1000 only | 0.043 / 0.078 | −0.009 | 0.06 | | 0.62 / 0.83 | 1.01 | 19.6 |
| + mixed only, + restore only | = production | | | | | 1.00 | 13.8, 25.5 |
| + peeliter 2 only | = production | | | | | 1.01 | 11.7 |
| full new (sf 1000, restore, mixed, warm start, peeliter 2) | 0.011 / 0.061 | −0.002 | 0.10 | 0.61 | 0.55 / 0.81 | 1.065 | 16.7 |
| full, but smooth_flux 0 | −0.008 / 0.019 | −0.002 | 0.34 | 0.42 | 0.28 / 0.44 | 0.99 | 16.3 |
| full + exempt 500 (recommended below 65 MHz) | −0.008 / 0.020 | −0.003 | 0.07 | | 0.35 / 0.57 | 1.00 | 15.2 |

| 55 MHz | Tau A | Cas A | Per B | Per A | 3C134 | 3C196 | blankV | s |
|---|---|---|---|---|---|---|---|---|
| production | −0.004 / 0.001 | −0.009 | 0.057 | 0.76 / 0.78 | 0.73 | 0.89 / 0.93 | 1.00 | 20.2 |
| full new | 0.002 / 0.007 | −0.003 | 0.014 | 0.87 / 0.94 | 0.85 | 0.94 / 0.98 | 1.11 | 12.1 |
| full, smooth_flux 0 | −0.003 / 0.001 | −0.003 | 0.069 | | 0.72 | 0.89 / 0.93 | 1.01 | 9.9 |
| full + exempt 500 | −0.003 / 0.001 | −0.003 | 0.003 | | 0.80 | 0.93 / 0.97 | 1.03 | 10.4 |

### 82 MHz (6 snapshots)

- Tau A and Cas A are given at 60 λ.
- blankV and blankI are at 60 λ.

| 82 MHz | Tau A | Cas A | blankV | blankI | s |
|---|---|---|---|---|---|
| production | 0.064 | 0.000 | 1.00 | 1.00 | 17.8 |
| full new | 0.277 | 0.008 | 1.18 | 1.17 | 10.6 |
| full, smooth_flux 0 (recommended above 65 MHz) | 0.064 | 0.000 | 1.01 | 1.01 | 8.7 |
| full, double / peeliter 3 / no warm start | 0.26–0.29 | 0.007–0.008 | 1.17–1.18 | 1.14–1.16 | |

### Findings

**smooth_flux is the only option that raises the Tau A residual and the V/I noise.**
- The cause is not Tau A's own smoothing: exempting Tau A changes nothing, and more iterations do not help.
- In production, per-channel ("sponge") gains of faint directions absorb Tau A's model residual and its V leakage.
- At 23 MHz the absorbing direction is Hya A. At 82 MHz it is the RFI direction plus Per B; smoothing either one alone raises Tau A from 0.06 to 0.16.
- The same sponges eat faint sources, so below about 60 MHz this is a trade-off. At 82 MHz smoothing gives no faint-source gain at all.

**Exempting sources of 500 Jy and up** keeps per-channel gains for Hya A, Tau A and the bright sources.
- This restores production's Tau A, Cas A and V noise at 23 and 55 MHz, and keeps most of the 55 MHz faint-source gain.
- At 23 MHz the 3C134 gain is lost: it came from smoothing Hya A.

**Mixed precision and peeliter 2** do not change the results measurably and save time. The sky_rfi engine with production options is bitwise-equivalent in every metric.

**Warm start** is harmless over 5-snapshot chains. Over 48 consecutive warm snapshots (23 MHz), compared with cold peeling of the same snapshots:
- blank V +10%;
- Tau A 0.065 → 0.098;
- Hya A left in (0.69 → 0.98);
- Per A 0.81 → 0.60.

Production chains are hundreds of snapshots long, so leave warm start off.

**"Per B" in the image comparison was Per A (3C84),** an unpeeled check source: the new config keeps it better. The catalogue "Per B" is 3C123, and it is removed better by the new config.

## Jupiter check (2024-12-20 / 12-27 burst snapshots)

Data: the earlier sponge study's prepped snapshots (`/lustre/nkosogor/peel_exp/runs/big1220|big1227/prep`).
- 2024-12-20: 18, 23 and 27 MHz at 03h and 12h, plus 18 MHz at 08h.
- 2024-12-27: 18, 23 and 27 MHz at 03h.
- Each night's restore JSON is in the same directory.
- CPU, real daemon, single snapshots, so no warm start applies.

**Jupiter position:** the |V| peak within 1° of the JSON position, on the unpeeled data. It lies 3–43′ from the ephemeris.

**Validation:** the unpeeled V at that position agrees with the study's image-based numbers:

| snapshot | this estimator | study |
|---|---|---|
| 18 MHz 03h | 89.2 kJy | 88.3 kJy |
| 23 MHz 03h | 35.8 kJy | 36.4 kJy |
| 27 MHz 03h | 1035 Jy | 1064 Jy |
| 12-27, 23 MHz | 928 Jy | 923 Jy |

The study's MSes had the stored UVW rotated by 0.2–0.35°, so fix_uvw was applied to the copies first. The peeler uses geometric UVW, so this does not change peeling.

### Jupiter Stokes V kept, peeled / unpeeled at 10 λ

There are 8 detections (|V| > 10 × the blank-sky V rms):

| configuration | median | range |
|---|---|---|
| production | 0.05 | −0.00 – 0.18 |
| full new (smooth 1000 + restore) | 0.995 | 0.969 – 1.097 |
| **recommended** (smooth 1000 + exempt 500 + restore, gate 65, cold) | **0.995** | **0.969 – 1.057** |
| recommended without restore | 0.10 | −0.00 – 0.27 |
| smooth 1000 + exempt 4000 + restore | 0.986 | 0.969 – 1.097 |

The 60 λ values are the same within ±0.02. Jupiter Stokes I kept is 0.84–0.99 with the restore, against 0.00–0.15 in production.

- The restore is what keeps Jupiter: it fits 600 Jy to 205 kJy and finds the V peak 2–15′ from the ephemeris.
- Exempting the ≥ 500 Jy directions does not cost Jupiter anything. The bright per-channel directions do absorb it (0.10 without restore), but the restore puts it back.
- Every configuration with restore keeps ≥ 0.959 of Jupiter's V on every detection.

### Other sources in these snapshots

- **03h burst snapshots:** Tau A, 3C196 and the blank-sky noise at 03h are dominated by Jupiter's own sidelobes (Jupiter is 9.6° from Tau A and up to 89 kJy). The ratios there measure the kept Jupiter, not the peeling.
- **12h: the exemption is LST-dependent.** Tau A is at 31° elevation (~600 Jy apparent) and Hya A at 39°. There smooth 1000 + exempt 500 leaves more Tau A than smooth 1000 without exemption:
  - 23 MHz: 0.26 vs 0.05 (production 0.04);
  - 27 MHz: 0.19 vs 0.13 (production 0.04).
- **18 MHz 12h Cas A (8° elevation):** −0.10 with exemption, against −0.20 without and −0.06 in production.
- **Trade-off:** at 10-31 05h (Tau A at 73°, Hya A at 12–22°) the exemption is what fixes Tau A (0.043 → −0.008). Catalog-flux exemption cannot tell these cases apart.
- **Possible improvement (not implemented):** an exemption by *apparent* (beam-weighted) flux or elevation in TTCalX, or an explicit source list.
- **Until then:** use exempt 500 when Tau A is high and Hya A is low, as in the hours the ablation covers. Otherwise smooth 1000 without exemption (or `smooth_exempt_flux: 4000`, which here is the same) is the more conservative choice below 65 MHz.

**Note on the 10-31 ablation:** Jupiter's position there was computed from the astropy ephemeris converted to barycentric ICRS *with* its distance. That is about 3° off (parallax). The ablation's Jupiter column is therefore not valid; Jupiter was weak that hour anyway (the restore fitted 3.5–6.5 Jy). The estimator now uses the direction only, plus the V-peak search above.
