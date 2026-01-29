# imaging_pipeline.py  –  v3  (with split after uvsub)
"""Imaging pipeline with model subtraction and peeling.

Implements a complete imaging workflow including:
- Calibration application (delay + bandpass)
- Antenna flagging
- TTCal peeling
- WSClean imaging (dirty + phase-centered)
- Source blanking and uvsub

This version includes split after uvsub for cleaner output.
"""
from __future__ import annotations
from pathlib import Path
import shutil, subprocess, logging, argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from casatasks import applycal, ft, uvsub, split        # type: ignore
from casatools import image                              # type: ignore
from orca.transform.flagging import flag_ants
from orca.wrapper.wsclean import wsclean
from orca.wrapper.change_phase_centre import change_phase_center

# -----------------------------------------------------------------------------#
#  STATIC CONFIG                                                               #
# -----------------------------------------------------------------------------#
DELAY_TABLE  = "/lustre/pipeline/calibration/delay/2024-05-24/20240524_delay.delay"
BANDPASS_TBL = "/lustre/pipeline/calibration/bandpass/{freq}/2024-05-24/11/bandpass_concat.{freq}_11.bandpass"

BAD_CORR_NUMS = (
    3, 12, 14, 17, 28, 31, 33, 34, 41, 44, 51, 56, 57, 79, 80, 87, 92,
    117, 124, 126, 127, 137, 150, 154, 178, 193, 201, 208, 211, 215, 218,
    224, 230, 231, 236, 242, 246, 261, 282, 294, 301, 307, 309, 311, 331,
)

JULIA_BIN    = "/opt/devel/pipeline/envs/julia060/bin/julia"
TTCAL_SCRIPT = "/opt/devel/pipeline/envs/julia060/bin/ttcal.jl"
SOURCES_JSON = os.path.expanduser("~/sources.json")

WSCLEAN_DIRTY = [
    "-pol", "IV", "-size", "4096", "4096",
    "-scale", "0.03125",
    "-niter", "20000",
    "-weight", "briggs", "0", "-horizon-mask", "10deg",
    "-taper-inner-tukey", "30",
]
WSCLEAN_PHASEC = [
    "-pol", "IV", "-size", "48", "48",
    "-scale", "0.0833333",
    "-niter", "1000",
    "-beam-size", "15amin",
    "-weight", "briggs", "0", "-horizon-mask", "10deg",
    "-taper-inner-tukey", "30",
]

# bright source to blank
SRC_RA_DEG, SRC_DEC_DEG = 219.750875, 64.291661
BLANK_SIZE_DEG = 0.5

# -----------------------------------------------------------------------------#
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
LOG = logging.getLogger("imaging_pipeline")

# -----------------------------------------------------------------------------#
#  HELPERS                                                                     #
# -----------------------------------------------------------------------------#
def copy_ms(src: Path, dst_dir: Path) -> Path:
    dst = dst_dir / src.name
    if dst.exists():
        LOG.info("[1] MS already copied → %s", dst)
    else:
        LOG.info("[1] Copying MS to workdir …")
        shutil.copytree(src, dst, dirs_exist_ok=True)
    return dst

# --- helper ------------------------------------------------------------------
def find_freq(ms_path: Path) -> str:
    """Return the first path component like '73MHz'."""
    for part in ms_path.parts:
        if part.endswith("MHz"):
            return part
    raise ValueError("Cannot infer frequency from path: " + str(ms_path))

# --- tweak apply_calibration() ----------------------------------------------
def apply_calibration(ms: Path, freq: str) -> None:
    tables = [
        DELAY_TABLE,
        BANDPASS_TBL.format(freq=freq),
    ]
    LOG.info("[2] applycal with %s", tables)
    applycal(vis=str(ms), gaintable=tables, calwt=[False], flagbackup=True)
    LOG.info("[2] applycal done.")


def run_flagging(ms: Path) -> None:
    LOG.info("[3] Flagging %d bad correlators …", len(BAD_CORR_NUMS))
    flag_ants(str(ms), BAD_CORR_NUMS)
    LOG.info("[3] Flagging done.")

def run_ttcal_zest(ms: Path) -> None:
    cmd = [
        JULIA_BIN, TTCAL_SCRIPT, "zest", str(ms), SOURCES_JSON,
        "--beam", "constant", "--minuvw", "10",
        "--maxiter", "5", "--tolerance", "1e-4",
    ]

    LOG.info("[4] ttcal zest …")
    LOG.info("Running command: %s", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    LOG.debug(res.stdout)
    if res.returncode:
        LOG.error(res.stderr); raise RuntimeError("ttcal failed")
    LOG.info("[4] ttcal completed.")

def wsclean_run(ms: Path, out_dir: Path, prefix: str, args: list[str], tag: str):
    LOG.info("[%s] WSClean %s", tag, ' '.join(args))
    wsclean(ms_list=[str(ms)], out_dir=str(out_dir), filename_prefix=prefix,
            extra_arg_list=args, num_threads=4, mem_gb=50)
    LOG.info("[%s] WSClean finished.", tag)

def plot_dirty(dirty_fits: Path, png_out: Path):
    with fits.open(dirty_fits) as h:
        data = h[0].data[0, 0]; wcs = WCS(h[0].header, naxis=[1, 2])
    plt.figure(figsize=(20, 15))
    ax = plt.subplot(projection=wcs)
    ax.imshow(data, cmap="jet", vmin=-15, vmax=95, origin="lower")
    ax.get_coords_overlay("fk5").grid(color="white", ls="dotted")
    plt.colorbar().set_label("Intensity")
    plt.savefig(png_out, dpi=150, bbox_inches="tight"); plt.close()
    LOG.info("[5] Dirty PNG → %s", png_out)


def blank_source(model_fits: Path, out_casa: Path):
    w = WCS(fits.getheader(model_fits))
    half = int((BLANK_SIZE_DEG / abs(w.wcs.cdelt[0])) // 2)
    x0, y0 = (int(_) for _ in w.wcs_world2pix([[SRC_RA_DEG, SRC_DEC_DEG, 0, 0]], 0)[0][:2])

    ia = image()
    if out_casa.exists():
        ia.open(str(out_casa))                      # reuse existing CASA img
        LOG.info("[6] Re-using CASA model → %s", out_casa)
    else:
        ia.fromfits(outfile=str(out_casa),          # <── keyword order fixed
                    infile=str(model_fits),
                    overwrite=True)
        ia.close(); ia.open(str(out_casa))

    data = ia.getchunk()
    data[x0-half:x0+half+1, y0-half:y0+half+1, :, :] = 0
    ia.putchunk(data); ia.done()
    LOG.info("[6] Source blanked at pix (%d,%d) ±%d", x0, y0, half)




def subtract_and_split(ms_src: Path, casa_model: Path, workdir: Path) -> Path:
    """Returns path to *_UVSUB_CORR.ms* (split, corrected column)."""
    uvsub_ms  = workdir / (ms_src.stem + "_UVSUB.ms")
    split_ms  = workdir / (ms_src.stem + "_UVSUB_CORR.ms")
    if split_ms.exists():
        LOG.info("[7] Re-using split MS → %s", split_ms); return split_ms

    LOG.info("[7] Copy → %s", uvsub_ms)
    shutil.copytree(ms_src, uvsub_ms)
    ft(vis=str(uvsub_ms), model=str(casa_model), usescratch=True)
    uvsub(vis=str(uvsub_ms))
    LOG.info("[7] uvsub done; splitting corrected data …")
    split(vis=str(uvsub_ms), outputvis=str(split_ms), datacolumn="corrected")
    LOG.info("[7] split produced → %s", split_ms)
    return split_ms

def rephase_center(ms: Path, ra_deg: float, dec_deg: float):
    centre = SkyCoord(ra_deg*u.deg, dec_deg*u.deg).to_string("hmsdms")
    LOG.info("[8] Rephasing to %s …", centre)
    change_phase_center(str(ms), centre)
    LOG.info("[8] Phase-centre shift done.")

# -----------------------------------------------------------------------------#
#  MAIN DRIVER                                                                 #
# -----------------------------------------------------------------------------#
def process_ms(ms_path: str | Path, workdir: str | Path) -> Path:
    ms_path  = Path(ms_path).expanduser().resolve()
    freq    = find_freq(ms_path)  
    workdir  = Path(workdir).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    copied_ms = copy_ms(ms_path, workdir)
    apply_calibration(copied_ms, freq)
    run_flagging(copied_ms)
    run_ttcal_zest(copied_ms)

    prefix_dirty = copied_ms.stem + "_20000_iterations"
    wsclean_run(copied_ms, workdir, prefix_dirty, WSCLEAN_DIRTY, "5")
    #plot_dirty(workdir/f"{prefix_dirty}-I-dirty.fits",
     #          workdir/f"{prefix_dirty}-dirty.png")

    #model_fits = workdir/f"{prefix_dirty}-I-model.fits"
    #casa_model = workdir/f"{prefix_dirty}-I-model.image"
    #blank_source(model_fits, casa_model)

    #split_ms = subtract_and_split(copied_ms, casa_model, workdir)
    #rephase_center(split_ms, SRC_RA_DEG, SRC_DEC_DEG)

    #prefix_final = split_ms.stem + "_1000_iterations_phase_center_small_15aminbeam"
    #wsclean_run(split_ms, workdir, prefix_final, WSCLEAN_PHASEC, "9")

    #final_fits = workdir/f"{prefix_final}-I-image.fits"
    #LOG.info("[10] Pipeline finished ✓ Final image → %s", final_fits)
    return copied_ms
    #return final_fits

# -----------------------------------------------------------------------------#
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="OVRO-LWA imaging pipeline")
    ap.add_argument("--ms",      required=True, help="Path to averaged MS")
    ap.add_argument("--workdir", required=True, help="Scratch directory")
    args = ap.parse_args()
    process_ms(args.ms, args.workdir)

