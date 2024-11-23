"""As in tau boo search
"""
from os import path
from glob import glob


from astropy.coordinates import SkyCoord
from astropy.wcs import wcs
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import warnings
from datetime import datetime

from orca.utils.coordutils import TAU_BOO, get_altaz_at_ovro
from orca.utils import fitsutils
from orca.transform import photometry

def vis(fn, lim=5):
    plt.figure()
    im, header = fitsutils.read_image_fits(fn)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cutout = fitsutils.get_cutout(im, TAU_BOO, wcs.WCS(header), 100)
    plt.imshow(cutout,vmin=-lim, vmax=lim, origin='lower')
    plt.text(0, 4, path.basename(fn), color='white')

def search(prefix: int, rms_search_size: int, peak_search_size: int, snr_threshold: float, lim=5):
    fns = sorted(glob(f'{prefix}/*/*/*fits'))
    peaks = []
    rmss = []
    for fn in fns:
        pk, n = photometry.search_src(fn, TAU_BOO, rms_search_size, peak_search_size)
        peaks.append(pk)
        rmss.append(n)
    peak_arr = np.array(peaks)
    rms_arr = np.array(rmss)
    snr_arr = np.abs(peak_arr) / rms_arr
    for i, fn in enumerate(fns):
        if snr_arr[i] > 6:
            vis(fn, lim=lim)


def make_movie(prefix: str, cutout_width=200, lim=5):
    plt.rcParams['animation.ffmpeg_path'] = '/opt/ffmpeg-7.0.2-amd64-static/ffmpeg'


def fn_to_beam(fn):
    fn = fn.split('/')[-1]
    ts = fn.split('.')[0]
    datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S")
    altaz = get_altaz_at_ovro(TAU_BOO, ts)
    alt = altaz.alt
    if alt.to(u.deg).value < 30:
        return np.nan
    return np.sin(alt.to(u.radian).value)**1.6

def fns_to_beams(fns):
    beams = []
    for fn in fns:
        b = fn_to_beam(fn)
        beams.append(b)
    return np.array(beams)

def rms_from_search(prefix: int, rms_search_size: int, peak_search_size: int, snr_threshold: float, lim=5):
    fns = sorted(glob(f'{prefix}/*/*/*fits'))
    peaks = []
    rmss = []
    for fn in fns:
        pk, n = photometry.search_src(fn, TAU_BOO, rms_search_size, peak_search_size)
        peaks.append(pk)
        rmss.append(n)
    peak_arr = np.array(peaks)
    rms_arr = np.array(rmss)
    beams = fns_to_beams(fns)
    rms_c = (rmss / beams)[~np.isnan(beams)]
    return rms_c

if __name__ == '__main__':
    fns = sorted(glob('/lustre/celery/V.fcoadds.fits/55MHz/*/*/*fits'))
    photometry.make_fig(fns, 
         TAU_BOO, 200, '/fast/yuping/15min_55MHz')