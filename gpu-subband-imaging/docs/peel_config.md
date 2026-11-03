# Peel configuration (sky_rfi engine): recommended settings and the ablation behind them

> **Status (2026-11-02).**
> - In `main`: every switch here (`engine: sky_rfi`, `precision`, `peeliter`, `warm_start`, `smooth_*`, `restore*`) in `PeelParams` and `workers/zest_daemon.jl`; all off by default (= v2).
> - Not in production: v2 uses the daemon engine, and the validated v3 setup (README) uses `sky_rfi` + `mixed` with no smoothing and no restore. The "Recommended" section below is this study's result, not the production config.
> - The rest is the CPU study record of 2026-11-01 (ablation, rule choice, held-out and Jupiter checks).
> - `smooth_exempt_below_el_deg` needs TTCalX v0.3.0 (`zest_sky_rfi!(...; smooth_exempt_names)`); on an older TTCalX the daemon exits with `CONFIG_ERROR`.

> **How far to trust the 30° rule:**
> - The threshold was chosen on 36 snapshots and checked on 26 held-out snapshots (new LST hours, a new night-hour, a new band).
> - It is the best rule tried. It is not robust everywhere: in 4 of 14 held-out band/hours, Tau A's residual is more than 0.03 above production.
> - In those cases every smoothing rule fails. The worst are 10-31 23 MHz 13h (0.35 vs 0.05) and 41 MHz 03h (0.45 vs 0.19).
> - See "Held-out check" below and the strict alternative.

This note is for the `peel:` section of `pipeline.yaml` with `engine: sky_rfi` (TTCalX `zest_sky_rfi!`).

- All measurements were made on CPU on 2026-11-01:
  - the option ablation: 2026-10-31 LST 05h at 23, 55 and 82 MHz;
  - the exemption-rule study: 36 snapshots;
  - the held-out check: 26 snapshots;
  - the Jupiter check: 2024-12-20 and 12-27 bursts.

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
  smooth_exempt_below_el_deg: 30.0   # sources below 30 deg (daemon elevation, this snapshot) keep per-channel
                                     # gains; Tau A can still be left at 0.2-0.45 in some hours (held-out check)
  smooth_below_mhz: 65.0             # subband centre frequency; 64 MHz band (centre ~66.4 MHz) and up: off
  # Jupiter restore below 41 MHz (one JSON per night, position from the ephemeris)
  restore: /path/to/jupiter_<date>.json
  restore_below_mhz: 41.0
  restore_offset_deg: 0.5
```

`peel_overrides` can do the same as the frequency gate, for example `{64: {smooth_flux: 0}, 69: ..., 82: ...}`.
The gate is simpler. It is logged on every snapshot as `PEEL_SMOOTH_GATE centre_mhz=... below_mhz=... smoothing=on|off` (`off` also when smooth_flux is 0).

**The 65 MHz gate is interpolated:**
- Measured: 55 MHz (smoothing helps faint sources) and 82 MHz (smoothing only hurts).
- Not measured: 59, 64 and 69 MHz.

**Strict alternative.** If Tau A must never be left in more than in production, use `smooth_flux: 0` everywhere, keeping mixed, peeliter 2, warm_start false and the restore.
- At 82 MHz this is equal to production in every metric.
- It gives up the faint-source gain: in the rule study 3C134 kept 0.66, instead of 0.82 with smoothing (all 36 snapshots).
- The restore still saves Jupiter without smoothing: Stokes V kept 0.986 (0.970–1.035) on the 8 bursts (Jupiter check below).

## The switches

All four are sent to the daemon only when set, so the defaults leave the daemon environment and its output unchanged.

| key | default | meaning |
|---|---|---|
| `smooth_below_mhz` | 0 (off) | apply `smooth_flux` and `rfi_smooth_flux` only when the subband **centre** (the mean channel frequency) is below this value. Note that `restore_below_mhz` instead tests the **lowest** channel: the restore runs unless the whole band is above it |
| `smooth_exempt_flux` | 0 (off) | sky sources whose catalog Stokes I at the band centre is at least this value keep per-channel gains. This is TTCalX `smooth_exempt_flux`, where 0 maps to `Inf` |
| `rfi_smooth_flux` | unset (= `smooth_flux`) | smoothing flux for the RFI emitters (TTCalX `rfi_smooth_flux`) |
| `smooth_exempt_below_el_deg` | 0 (off) | sky sources below this elevation in the snapshot keep per-channel gains. The daemon builds the name list per snapshot from the same elevations it uses for `min_source_elevation_deg`, passes it as TTCalX `smooth_exempt_names`, and logs it as `PEEL_SMOOTH_EXEMPT below_el_deg=... names="..."` |

- Every value must be at least 0. A negative value raises `ConfigError`, and the daemon exits with `CONFIG_ERROR`.
- The `SKY_RFI` settings line shows all four and reaches the `gsi.peel` log.
- **Elevation definition.** The daemon (and `min_source_elevation_deg`) uses asin(n) of the source direction about the MS phase centre.
  - In the LST-centred snapshots this is within ±0.2° of the true altitude (median +0.05°, 157 source/snapshot pairs).
  - All thresholds and elevations quoted below are these daemon elevations (PEEL_SOLVE `elevation_deg`).
- In the 7-source model, catalog flux at about 25 MHz is: Cas A and Cyg A above 10 kJy, Vir A about 7 kJy, Tau A about 3.2 kJy, Hya A and Her A about 600 Jy, Per B (3C123) 400 Jy.
- `smooth_exempt_flux: 500` therefore leaves only Per B and the RFI emitters band-tied. That rule is LST-dependent and is **not** recommended; it is kept as a switch.

Checks, on CPU through the real daemon:
- 82 MHz with `smooth_below_mhz: 65` is bitwise equal to `smooth_flux: 0` (DATA and FLAG, 2 snapshots).
- The default settings give bitwise the same output as the previous daemon (82 MHz new config; 23 MHz production).
- 23 MHz with `smooth_exempt_flux: 500` reproduces the ablation run E8 to 3e-8 Jy.
- `smooth_exempt_below_el_deg: 30` through `PeelParams` reproduces the study's rule L30 exactly: 0.0 Jy difference over 2 × 2016 values (10-31 23 MHz × 12; 12-20 × 12).
- With the new code and its defaults, both the conservative and the production configs are bitwise equal to the previous commit's daemon.

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
| full + catalog exempt 500 (superseded, see the rule study) | −0.008 / 0.020 | −0.003 | 0.07 | | 0.35 / 0.57 | 1.00 | 15.2 |

| 55 MHz | Tau A | Cas A | Per B | Per A | 3C134 | 3C196 | blankV | s |
|---|---|---|---|---|---|---|---|---|
| production | −0.004 / 0.001 | −0.009 | 0.057 | 0.76 / 0.78 | 0.73 | 0.89 / 0.93 | 1.00 | 20.2 |
| full new | 0.002 / 0.007 | −0.003 | 0.014 | 0.87 / 0.94 | 0.85 | 0.94 / 0.98 | 1.11 | 12.1 |
| full, smooth_flux 0 | −0.003 / 0.001 | −0.003 | 0.069 | | 0.72 | 0.89 / 0.93 | 1.01 | 9.9 |
| full + catalog exempt 500 (superseded) | −0.003 / 0.001 | −0.003 | 0.003 | | 0.80 | 0.93 / 0.97 | 1.03 | 10.4 |

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

**Exempting sources of 500 Jy and up (catalog)** keeps per-channel gains for Hya A, Tau A and the bright sources.
- At 10-31 05h this restores production's Tau A, Cas A and V noise at 23 and 55 MHz, and keeps most of the 55 MHz faint-source gain.
- At other LSTs it leaves more Tau A (next section), so the elevation rule replaced it.

**Mixed precision and peeliter 2** do not change the results measurably and save time. The sky_rfi engine with production options is bitwise-equivalent in every metric.

**Warm start** is harmless over 5-snapshot chains. Over 48 consecutive warm snapshots (23 MHz), compared with cold peeling of the same snapshots:
- blank V +10%;
- Tau A 0.065 → 0.098;
- Hya A left in (0.69 → 0.98);
- Per A 0.81 → 0.60.

Production chains are hundreds of snapshots long, so leave warm start off.

**"Per B" in the image comparison was Per A (3C84),** an unpeeled check source: the new config keeps it better. The catalogue "Per B" is 3C123, and it is removed better by the new config.

## Choosing the exemption rule (tuning set, 36 snapshots)

### Data and variants

- 36 snapshots on CPU, all cold:
  - 2026-10-31 LST 05h: 23 MHz × 12 and 55 MHz × 6;
  - 2024-12-20: 03, 08 and 12h at 18, 23, 27 and 55 MHz;
  - 2024-12-27: 03 and 12h at 18, 23 and 27 MHz.
- All variants use the sky_rfi engine (except production) with mixed precision, peeliter 2, warm start off, restore below 41 MHz and the 65 MHz gate:

| variant | rule |
|---|---|
| **P** | production |
| **C** | conservative: smooth 1000, nothing exempt |
| **R500** | catalog flux ≥ 500 Jy exempt |
| **A1000** | apparent flux ≥ 1000 Jy exempt, catalog × sin(el)^1.6 |
| **L20, L30, L35** | sources below 20 / 30 / 35° elevation exempt |

- The sin(el)^1.6 is a fit to the OVRO-LWA Stokes I beam (`OVRO-LWA_MROsoil_updatedheight.h5`, via orca): exponent 1.6 at 20–25 MHz, 1.5 at 55 MHz, 1.2 at 84 MHz.
- [`peel_config_ratios.csv`](peel_config_ratios.csv) has the group medians for production / conservative / recommended (L30) / strict, per set, night, band and hour.
  - It covers all 35 groups: 20 tuning, 14 held-out, and 82 MHz from the option ablation.
  - Ratios are given signed and as |·|, at 10 and 60 λ, with `held_out` and `jupiter_detected` flags.
  - At 82 MHz the 65 MHz gate turns smoothing off, so conservative = recommended = strict there (bitwise, see the checks above).
- [`peel_config_fig_bands.csv`](peel_config_fig_bands.csv) is the figure table: per band (18, 23, 27, 41, 55, 82 MHz), quantity and config, the median and 16–84% range over that band's groups.
  - Residuals are |·| at 60 λ (Tau A, Cas A, Hya A, 3C123). Retained sources are at 10 λ (3C134, 3C196, Per A, 3C48). Both are given over all groups and over non-Jupiter groups.
  - Jupiter V kept is over Jupiter-detected snapshots; blank-sky V and I relative to production are over non-Jupiter groups.
  - Strict was run on every tuning and held-out snapshot (gsi daemon 00871ad, TTCalX c78f0b5; both in `main` / v0.3.0). The new production / conservative / L30 values reproduce the earlier table exactly.

**Jupiter-affected groups are excluded from the medians.** Jupiter is detected in 8 of the 20 groups:
- 12-20 18/23/27 MHz 03h;
- 12-20 18/23 MHz 12h;
- 12-27 18/23/27 MHz 03h.

There the kept Jupiter's sidelobes dominate Tau A, 3C196 and the blank-sky noise.

### Medians over the 12 groups without Jupiter

- Residuals are given as |·| (signed values reward over-subtraction); the signed Tau A median is shown alongside.
- Check sources are kept fractions at 10 λ.

| | \|Tau A\| 60 λ, median / worst | signed Tau A 60 λ | \|Tau A\| 10 λ | groups with \|Tau A\| > prod + 0.03 | \|Cas A\| | \|Hya A\| | 3C134 | 3C196 | Per A | blank V / prod |
|---|---|---|---|---|---|---|---|---|---|---|
| P | 0.029 / 0.115 | 0.029 | 0.030 | – | 0.0020 | 0.25 | 0.852 | 0.933 | 0.32 | 1.000 |
| C | 0.079 / 0.451 | 0.079 | 0.080 | 6 | 0.0016 | 0.12 | 0.872 | 0.964 | 0.68 | 1.051 |
| R500 | 0.075 / 0.390 | 0.075 | 0.095 | 5 | 0.0019 | 0.23 | 0.858 | 0.952 | 0.46 | 1.039 |
| A1000 | 0.079 / 0.451 | 0.079 | 0.079 | 6 | 0.0017 | 0.12 | 0.870 | 0.964 | 0.67 | 1.051 |
| L20 | 0.036 / 0.451 | 0.020 | 0.038 | 4 | 0.0016 | 0.12 | 0.865 | 0.954 | 0.65 | 1.023 |
| **L30** | **0.019** / 0.451 | 0.015 | **0.009** | **2** | 0.0016 | 0.23 | 0.863 | 0.950 | 0.47 | **1.015** |
| L35 | 0.023 / 0.390 | 0.021 | 0.014 | 2 | 0.0017 | 0.23 | 0.862 | 0.950 | 0.47 | 1.015 |

### Tau A residual at 60 λ, P / C / R500 / L30 (signed; daemon elevations)

| group | P | C | R500 | L30 | sources exempt under L30 |
|---|---|---|---|---|---|
| 10-31 23 MHz 05h | 0.021 | 0.068 | 0.022 | 0.025 | Hya A (12–22°); Cas A once it is below 30° |
| 10-31 55 MHz 05h | 0.001 | 0.009 | 0.000 | 0.001 | Hya A |
| 12-20 23 MHz 08h | 0.062 | 0.090 | 0.064 | 0.064 | Cas A 26°, Vir A, Hya A 23° |
| 12-20 27 MHz 12h | 0.037 | 0.130 | 0.192 | 0.017 | Cas A, Per B 23.5° (Tau A at 30.75°) |
| 12-20 55 MHz 03h | 0.071 | 0.451 | 0.390 | 0.451 | none (Tau A at 30.17°) |
| 12-20 55 MHz 12h | 0.014 | 0.127 | 0.225 | 0.013 | Cas A, Per B |
| 12-27 23 MHz 12h | 0.064 | 0.224 | 0.323 | 0.020 | Cas A, Tau A 25.3°, Per B |
| 12-27 27 MHz 12h | 0.067 | 0.192 | 0.323 | 0.102 | Cas A, Tau A, Per B |

**Why elevation helps.** What lowers Tau A's residual is per-channel freedom in the poorly modelled directions, and those are the low ones:
- Hya A at 05h;
- Per B and Cas A at 12h.

Catalog and apparent flux pick the bright, high sources instead (Tau A and Hya A at 12h, Tau A and Cas A at 05h).

**Caveats:**
- The threshold was tuned on these snapshots.
- At 12-20 03h and 12h Tau A sits at 30.17° and 30.75°, just above 30°, so it is not exempt. With 35° it is exempt: 55 MHz 03h improves a little (0.45 → 0.39), and 12-20 12h gets worse (0.011 → 0.035 at 23 MHz).
- 20° misses Per B at 23.5°.

## Held-out check (26 snapshots not used for tuning)

### Data

- 2026-10-31 at 12 and 13 UTC (LST about 6.8–8.3 h; before sunrise): 23 and 55 MHz, 4 snapshots per hour.
  - Prepared from the read-only slow-buffer tars with the same v2 prep (bp 2026-10-31 21h, badants, chanbin 4, fix_uvw).
  - Daemon elevations: Tau A 68/57°, Cas A 21/16°, Hya A 30/37°, Vir A low.
- 2024-12-27 08h (an unused night/hour) at 18, 23, 27, 55 and 41 MHz: Cas A 23°, Hya A 27°.
- 41 MHz (an unused band) at 2024-12-20 and 12-27, 03/08/12h: Tau A 25–36°, Cyg A 28–33°.
- There is no Jupiter detection in the held-out set.

### Results

\|Tau A\| residual at 60 λ, group medians:

| group | P | C | L20 | **L30** | L35 |
|---|---|---|---|---|---|
| 10-31 23 MHz 12h | 0.020 | 0.058 | 0.054 | 0.046 | 0.046 |
| 10-31 23 MHz 13h | 0.049 | 0.352 | 0.352 | 0.348 | 0.347 |
| 10-31 55 MHz 12h | 0.003 | 0.007 | 0.006 | 0.004 | 0.004 |
| 10-31 55 MHz 13h | 0.005 | 0.008 | 0.008 | 0.007 | 0.007 |
| 12-20 41 MHz 03h | 0.192 | 0.454 | 0.454 | 0.454 | 0.532 |
| 12-20 41 MHz 08h | 0.005 | 0.035 | 0.020 | 0.014 | 0.014 |
| 12-20 41 MHz 12h | 0.037 | 0.178 | 0.178 | 0.022 | 0.067 |
| 12-27 18 MHz 08h | 0.011 | 0.023 | 0.011 | 0.004 | 0.004 |
| 12-27 23 MHz 08h | 0.002 | 0.035 | 0.020 | 0.009 | 0.009 |
| 12-27 27 MHz 08h | 0.001 | 0.037 | 0.014 | 0.007 | 0.007 |
| 12-27 41 MHz 03h | 0.016 | 0.204 | 0.204 | 0.204 | 0.204 |
| 12-27 41 MHz 08h | 0.010 | 0.053 | 0.021 | 0.015 | 0.015 |
| 12-27 41 MHz 12h | 0.026 | 0.213 | 0.019 | 0.061 | 0.061 |
| 12-27 55 MHz 08h | 0.004 | 0.007 | 0.008 | 0.007 | 0.007 |

Medians over the 14 held-out groups:

| | \|Tau A\| 60 λ (worst) | groups with \|Tau A\| > prod + 0.03 | \|Cas A\| (worst) | \|Cyg A\| | \|Hya A\| | 3C134 | 3C196 | Per A | blank V / prod | blank I / prod |
|---|---|---|---|---|---|---|---|---|---|---|
| P | 0.010 (0.192) | – | 0.0075 (0.018) | 0.004 | 0.50 | 0.760 | 0.870 | 0.751 | 1.000 | 1.000 |
| C | 0.045 (0.454) | 9 | 0.0059 (0.021) | 0.005 | 0.05 | 0.902 | 0.923 | 0.858 | 1.013 | 1.021 |
| L20 | 0.020 (0.454) | 5 | 0.0063 (0.021) | 0.005 | 0.05 | 0.899 | 0.922 | 0.848 | 1.023 | 1.019 |
| **L30** | **0.015** (0.454) | **4** | 0.0066 (0.020) | 0.005 | 0.41 | 0.863 | 0.914 | 0.811 | 1.026 | 1.019 |
| L35 | 0.015 (0.532) | 5 | 0.0066 (0.019) | 0.006 | 0.53 | 0.863 | 0.904 | 0.805 | 1.026 | 1.017 |

### Assessment

- L30 is again the best of the rules tried: the lowest \|Tau A\| median and the fewest regressions (4/14, against 9/14 without exemption).
- It keeps about half of the faint-source gain: 3C134 0.76 → 0.86 (C 0.90), Per A 0.75 → 0.81 (C 0.86).
- Blank-sky V and I rise by about 2–3%.
- **It is not robust everywhere.** In three held-out groups every smoothing rule leaves far more Tau A than production:
  - 10-31 23 MHz 13h: 0.35 vs 0.05 (Tau A at 57°; the only low source is Cas A at 16°);
  - 12-20 41 MHz 03h: 0.45 vs 0.19;
  - 12-27 41 MHz 03h: 0.20 vs 0.02 (Tau A at 30–36°).
- Together with 12-20 55 MHz 03h in the tuning set, these show that band-gain smoothing can leave 20–45% of Tau A whenever no low direction absorbs Tau A's model error. Which one does depends on the sky, not only on the threshold.
- So: use L30 if the faint-source gain is wanted and occasional Tau A residuals of 0.2–0.45 (about 1 hour in 4 at 23–55 MHz here) are acceptable. Otherwise use the strict alternative (`smooth_flux: 0`).
- **Possible next step, not implemented:** keep per-channel gains for every direction when the Tau A solve shows a large residual (an adaptive guard), or improve the Tau A model. Tau A is partly resolved on the longest baselines at 41–82 MHz.

## Jupiter check (2024-12-20 / 12-27 burst snapshots)

**Data:** the earlier sponge study's prepped snapshots, with each night's restore JSON. CPU, real daemon, single snapshots.

**Jupiter position:** the |V| peak within 1° of the JSON position, on the unpeeled data. It lies 3–43′ from the ephemeris.

**Validation:** the unpeeled V there agrees with the study's image-based values:

| snapshot | this estimator | study |
|---|---|---|
| 18 MHz 03h | 89.2 kJy | 88.3 kJy |
| 23 MHz 03h | 35.8 kJy | 36.4 kJy |
| 27 MHz 03h | 1035 Jy | 1064 Jy |
| 12-27, 23 MHz | 928 Jy | 923 Jy |

The study's MSes had the stored UVW rotated by 0.2–0.35°; fix_uvw was applied to the copies first. The peeler uses geometric UVW, so peeling is unchanged.

### Jupiter Stokes V kept, peeled / unpeeled at 10 λ

There are 8 detections (|V| > 10 × the blank-sky V rms) in the tuning set: 12-20 18/23/27 MHz 03h and 18/23 MHz 12h, 12-27 18/23/27 MHz 03h.

| configuration | median | range |
|---|---|---|
| production | 0.05 | −0.00 – 0.18 |
| conservative (smooth 1000 + restore) | 0.995 | 0.969 – 1.097 |
| **recommended L30** (smooth 1000 + below-30° exemption + restore) | **0.995** | **0.969 – 1.042** |
| **strict** (smooth_flux 0 + restore) | **0.986** | **0.970 – 1.035** |
| catalog exempt 500 + restore (superseded) | 0.995 | 0.969 – 1.057 |
| catalog exempt 500 without restore | 0.10 | −0.00 – 0.27 |

- The 60 λ values are the same within ±0.02. Jupiter Stokes I kept is 0.84–0.99 with the restore.
- The restore is what keeps Jupiter: per-channel directions absorb it (0.10 without restore), and the restore fits it back (600 Jy to 205 kJy, V peak 2–15′ from the ephemeris).
- Every rule with the restore keeps at least 0.969 of Jupiter's V on every detection at 10 λ.

### Strict vs L30 vs production on the 8 Jupiter detections

The run uses gsi daemon 00871ad with TTCalX c78f0b5 (both in `main` / v0.3.0). It reproduces the earlier L30 and production runs exactly (0 difference over 448 values each).

Medians, with ranges:

| | Jupiter V kept 10 λ | Jupiter V kept 60 λ | Jupiter I kept | \|Tau A\| 60 λ | \|Cas A\| 60 λ | 3C134 | Per A | blank V / prod | blank I / prod | peel s |
|---|---|---|---|---|---|---|---|---|---|---|
| production | 0.046 (−0.00–0.18) | 0.049 | 0.060 | 0.31 | 0.006 | 0.850 | 0.75 | 1.00 | 1.00 | 20.7 |
| L30 | 0.995 (0.969–1.042) | 0.988 (0.959–1.052) | 0.971 (0.85–0.99) | 0.50 | 0.005 | 0.934 | 0.87 | 3.2 | 1.12 | 19.2 |
| strict | 0.986 (0.970–1.035) | 0.978 (0.935–0.999) | 0.968 (0.85–0.99) | 0.28 | 0.005 | 0.918 | 0.84 | 3.3 | 1.09 | 18.2 |

- **The restore saves Jupiter without smoothing.** The 10 λ V is at least 0.97 on every detection. At 60 λ the lowest value is 0.935 (12-27 18 MHz 03h, 287 Jy).
- **These snapshots are dominated by the kept Jupiter:**
  - blank-sky V is up to 80× production at 18 MHz 03h, from Jupiter's own sidelobes;
  - the Tau A ratios there, at 9.6° from Jupiter, are not peeling metrics.
- Even so, strict leaves less at the Tau A position than L30 on 7 of the 8 detections. That is consistent with the held-out finding.
- **12-20 18 MHz 12h (Jupiter 364 Jy):** Cas A is 0.070 strict, 0.173 L30 and 0.064 production.

**Note on the 10-31 option ablation:** its Jupiter position came from the astropy ephemeris converted to barycentric ICRS *with* the distance, which is about 3° off (parallax). That column is not valid; Jupiter was weak that hour (the restore fitted 3.5–6.5 Jy). The estimator now uses the direction only, which matches the restore JSONs to 1.6′, plus the V-peak search.
