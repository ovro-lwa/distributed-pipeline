#!/usr/bin/env python3
"""RMS-vs-subband plot + wideband stacking + 3-colour composites.

Lightweight Phase 3 script — no email, no transient search, no photometry.
Run after all subbands for one LST hour have been archived to Lustre.

Usage
-----
# RMS plot + wideband stacking + 3-colour:
python pipeline/wideband_summary.py /lustre/pipeline/images/01h/2024-12-27/Run_YYYYMMDD

# RMS plot only (skip wideband stacking):
python pipeline/wideband_summary.py --rms-only /lustre/pipeline/images/01h/2024-12-27/Run_YYYYMMDD

Outputs (in <run_dir>/Wideband/):
  - thermal_noise.csv                    — freq, rms, weight columns
  - thermal_noise_vs_subband.png         — log-scale RMS vs frequency
  - Wideband_<Band>_<Pol>_<Cat>_*.fits   — inverse-variance co-adds
  - Wideband_*_3color.png               — RGB composites
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
from astropy.io import fits


# Wideband band definitions (same as post_process_science.py)
WIDEBAND_BANDS = {
    'Red':   (18, 41),
    'Green': (41, 64),
    'Blue':  (64, 85),
}


def get_inner_rms(fits_path: str) -> float:
    """RMS of the inner 25% (50% per side) of a FITS image."""
    with fits.open(fits_path) as hdul:
        data = hdul[0].data.squeeze()
        h, w = data.shape
        ch, cw = h // 2, w // 2
        rh, rw = h // 4, w // 4
        region = data[ch - rh:ch + rh, cw - rw:cw + rw]
        return float(np.nanstd(region))


def _stack_images(img_list, weights, out_path, history_str):
    """Inverse-variance weighted co-add of FITS images."""
    if not img_list:
        return False
    data_sum = None
    weight_sum = 0.0
    ref_header = None

    for img, w in zip(img_list, weights):
        with fits.open(img) as hdul:
            data = hdul[0].data.squeeze()
            if data_sum is None:
                data_sum = np.zeros_like(data, dtype=np.float64)
                ref_header = hdul[0].header.copy()
            if data.shape != data_sum.shape:
                continue
            data_sum += data * w
            weight_sum += w

    if weight_sum > 0:
        final_data = data_sum / weight_sum
        ref_header['BTYPE'] = 'Intensity'
        ref_header['HISTORY'] = history_str
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fits.writeto(out_path, final_data.astype(np.float32),
                     ref_header, overwrite=True)
        return True
    return False


def _make_3color_png(red_path, green_path, blue_path, out_png, title=""):
    """Generate a 3-colour PNG from Red/Green/Blue FITS images."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.ndimage import zoom as _zoom

    channels = []
    for fpath in [red_path, green_path, blue_path]:
        if fpath and os.path.exists(fpath):
            with fits.open(fpath) as hdul:
                channels.append(hdul[0].data.squeeze().astype(np.float64))
        else:
            channels.append(None)

    valid = [c for c in channels if c is not None]
    if len(valid) < 2:
        print(f"  3-colour PNG needs >=2 bands, got {len(valid)} — skipping")
        return

    ref_shape = max((c.shape for c in valid), key=lambda s: s[0] * s[1])
    for i in range(3):
        if channels[i] is None:
            channels[i] = np.zeros(ref_shape, dtype=np.float64)
        elif channels[i].shape != ref_shape:
            zoom_factors = (ref_shape[0] / channels[i].shape[0],
                            ref_shape[1] / channels[i].shape[1])
            channels[i] = _zoom(channels[i], zoom_factors, order=1)

    normed = []
    for ch in channels:
        vmin, vmax = np.nanpercentile(ch, [0.5, 99.8])
        if vmax <= vmin:
            vmax = vmin + 1
        clipped = np.clip(ch, vmin, vmax)
        normed.append((clipped - vmin) / (vmax - vmin))

    rgb = np.stack(normed, axis=-1)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    ax.imshow(rgb, origin='lower', interpolation='nearest')
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=14, color='white',
                     bbox=dict(facecolor='black', alpha=0.7))
    plt.savefig(out_png, dpi=150, bbox_inches='tight', facecolor='black')
    plt.close()
    print(f"  Saved 3-colour PNG: {os.path.basename(out_png)}")


def measure_noise(run_dir, out_dir):
    """Measure Stokes-V RMS per subband, save CSV + plot."""
    v_images = glob.glob(
        os.path.join(run_dir, "*MHz", "V", "deep", "*V-Taper-Deep*image*.fits"))
    v_images = [f for f in v_images
                if "pbcorr" not in f and "dewarped" not in f]

    if not v_images:
        print(f"No Stokes-V deep images found in {run_dir}/*MHz/V/deep/")
        return None

    print(f"Found {len(v_images)} Stokes-V deep images")

    rows = []
    for v_img in sorted(v_images):
        freq_str = v_img.split('/')[-4].replace('MHz', '')
        try:
            freq = float(freq_str)
        except ValueError:
            continue
        rms = get_inner_rms(v_img)
        if np.isfinite(rms) and rms > 0:
            rows.append({'freq': freq, 'rms': rms, 'weight': 1.0 / (rms ** 2)})
            print(f"  {freq:5.0f} MHz  RMS = {rms * 1000:8.2f} mJy/beam")

    if not rows:
        print("No valid RMS measurements.")
        return None

    df = pd.DataFrame(rows).sort_values('freq')
    csv_path = os.path.join(out_dir, "thermal_noise.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV: {csv_path}")

    # Plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(df['freq'], df['rms'] * 1000, 'o-',
                color='steelblue', markersize=6, linewidth=1.5)
    ax.set_xlabel('Frequency (MHz)', fontsize=12)
    ax.set_ylabel('Stokes V RMS (mJy/beam)', fontsize=12)
    ax.set_title('Thermal Noise vs Sub-band', fontsize=14)
    ax.grid(True, alpha=0.3)
    png_path = os.path.join(out_dir, "thermal_noise_vs_subband.png")
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot: {png_path}")

    print(f"\nMedian RMS: {df['rms'].median() * 1000:.2f} mJy/beam")
    print(f"Min RMS:    {df['rms'].min() * 1000:.2f} mJy/beam "
          f"(@ {df.loc[df['rms'].idxmin(), 'freq']:.0f} MHz)")

    return df


def run_wideband(run_dir, out_dir, df_noise):
    """Wideband inverse-variance stacking + 3-colour PNGs."""
    print("\n--- Wideband Stacking ---")

    stacking_targets = [
        ('I', 'deep',  'Taper', 'Robust-0-',   'Robust-0'),
        ('I', 'deep',  'NoTaper', 'Robust-0-',  'Robust-0'),
        ('I', 'deep',  'NoTaper', 'Robust-0.75-', 'Robust-0.75'),
        ('V', 'deep',  'Taper', 'Deep',         'Robust-0'),
    ]

    wideband_files = {}

    for b_name, (f_min, f_max) in WIDEBAND_BANDS.items():
        subset = df_noise[(df_noise['freq'] >= f_min) & (df_noise['freq'] < f_max)]
        if subset.empty:
            continue

        print(f"\n  {b_name} ({f_min}-{f_max} MHz, {len(subset)} sub-bands)")

        for pol, cat, taper_kw, suffix_kw, robust_label in stacking_targets:
            img_list, weights = [], []
            for _, row in subset.iterrows():
                freq_dir = os.path.join(
                    run_dir, f"{int(row['freq'])}MHz", pol, cat)
                if not os.path.isdir(freq_dir):
                    continue

                # Try dewarped first, fall back to raw
                candidates = glob.glob(
                    os.path.join(freq_dir, f"*{suffix_kw}*_dewarped*.fits"))
                candidates = [f for f in candidates
                              if "pbcorr" not in os.path.basename(f)
                              and "_dewarped_dewarped" not in os.path.basename(f)]
                if not candidates:
                    candidates = glob.glob(
                        os.path.join(freq_dir, f"*{suffix_kw}*image*.fits"))
                    candidates = [f for f in candidates
                                  if "pbcorr" not in os.path.basename(f)
                                  and "dewarped" not in os.path.basename(f)]

                if taper_kw == 'NoTaper':
                    candidates = [f for f in candidates
                                  if 'NoTaper' in os.path.basename(f)]
                else:
                    candidates = [f for f in candidates
                                  if 'NoTaper' not in os.path.basename(f)]

                if suffix_kw == 'Robust-0-':
                    candidates = [f for f in candidates
                                  if 'Robust-0.75' not in os.path.basename(f)]

                if candidates:
                    img_list.append(sorted(candidates)[0])
                    weights.append(row['weight'])

            if not img_list:
                continue

            taper_str = "Taper" if taper_kw != 'NoTaper' else "NoTaper"
            out_name = (f"Wideband_{b_name}_{pol}_{cat}_"
                        f"{taper_str}_{robust_label}.fits")
            out_path = os.path.join(out_dir, out_name)
            history = (f"Wideband {b_name} ({f_min}-{f_max} MHz) "
                       f"{pol} {cat} {taper_str} {robust_label}")
            if _stack_images(img_list, weights, out_path, history):
                print(f"    Stacked: {out_name} ({len(img_list)} images)")
                wideband_files[(b_name, pol, cat, taper_str, robust_label)] = out_path

    # 3-colour PNGs
    print("\n--- 3-Colour PNGs ---")

    # Mixed robust
    r = wideband_files.get(('Red', 'I', 'deep', 'NoTaper', 'Robust-0.75'))
    g = wideband_files.get(('Green', 'I', 'deep', 'NoTaper', 'Robust-0'))
    b = wideband_files.get(('Blue', 'I', 'deep', 'NoTaper', 'Robust-0'))
    if any(p is not None for p in [r, g, b]):
        _make_3color_png(r, g, b,
                         os.path.join(out_dir, "Wideband_I_deep_NoTaper_mixed_3color.png"),
                         title="Wideband I — Red:R-0.75 Green/Blue:R0")

    # Per-robust
    for robust_str in ['Robust-0', 'Robust-0.75']:
        r = wideband_files.get(('Red', 'I', 'deep', 'NoTaper', robust_str))
        g = wideband_files.get(('Green', 'I', 'deep', 'NoTaper', robust_str))
        b = wideband_files.get(('Blue', 'I', 'deep', 'NoTaper', robust_str))
        if any(p is not None for p in [r, g, b]):
            _make_3color_png(
                r, g, b,
                os.path.join(out_dir, f"Wideband_I_deep_NoTaper_{robust_str}_3color.png"),
                title=f"Wideband I NoTaper {robust_str}")

    return wideband_files


def main():
    parser = argparse.ArgumentParser(
        description="RMS vs sub-band + wideband stacking + 3-colour composites")
    parser.add_argument("run_dir",
                        help="Run directory containing per-subband folders")
    parser.add_argument("-o", "--output", default=None,
                        help="Output directory (default: <run_dir>/Wideband/)")
    parser.add_argument("--rms-only", action="store_true",
                        help="Only produce the RMS plot, skip wideband stacking")
    args = parser.parse_args()

    run_dir = args.run_dir
    out_dir = args.output or os.path.join(run_dir, "Wideband")
    os.makedirs(out_dir, exist_ok=True)

    df_noise = measure_noise(run_dir, out_dir)
    if df_noise is None:
        sys.exit(1)

    if not args.rms_only:
        run_wideband(run_dir, out_dir, df_noise)

    print("\nDone.")


if __name__ == "__main__":
    main()
