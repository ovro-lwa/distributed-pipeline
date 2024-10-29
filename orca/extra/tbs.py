from typing import Tuple

import numpy as np

from astropy.coordinates import SkyCoord, ICRS
import astropy.units as u
from astropy import wcs

from orca.utils import fitsutils
from orca.transform import photometry

def search_src(fns: str, src: SkyCoord, stats_box_size: int, peak_search_box_size: int) -> Tuple[float, float]:
    data, header = fitsutils.read_image_fits(fns)
    w = wcs.WCS(header)
    noise_cutout = fitsutils.get_cutout(data, src, w, stats_box_size // 2)
    rms = photometry.estimate_image_noise(noise_cutout)
    peak_cutout = fitsutils.get_cutout(data, src, w, peak_search_box_size // 2)
    peak = peak_cutout.max()
    return peak, rms

