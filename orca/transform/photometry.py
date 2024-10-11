from typing import Optional
import logging
import warnings

from astropy import wcs
from astropy.io import fits

from orca.utils import fitsutils
import numpy as np

logger = logging.getLogger(__name__)

def std_and_max_around_coord(fits_file, coord, radius=5):
    """
    Calculate the stdev around a given coordinate in a FITS file.

    Parameters
    ----------
    fits_file : str
        Path to the FITS file.
    coord : astropy.Coordinates
        The coordinate around which to calculate the stdev.
    radius : int, optional
        Radius in pixels.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        data, header = fitsutils.read_image_fits(fits_file)
    w = wcs.WCS(header)
    return fitsutils.std_and_max_around_src(data.T, radius, coord, w)

def average_with_rms_threshold(fits_list, out_fn, source_coord, radius_px, threshold_multiple) -> Optional[str]:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        contents = [fitsutils.read_image_fits(fn) for fn in fits_list]
        rms_arr = np.zeros(len(contents))
        for i, (data, header) in enumerate(contents):
            w = wcs.WCS(header)
            x, y = wcs.utils.skycoord_to_pixel(source_coord, w)
            if np.isnan(x) or np.isnan(y):
                raise ValueError(f'Coordinate {source_coord} is not in the image.')
            x = int(x)
            y = int(y)
            im_box = data.T[x - radius_px :x + radius_px, y - radius_px : y + radius_px]
            rms_arr[i] = np.std(im_box)
        rms_threshold = np.median(rms_arr[rms_arr > 0]) * threshold_multiple

        n = np.sum(rms_arr <= rms_threshold)
        out_data = np.zeros_like(contents[0][0])
        logger.info(f'{len(contents)-n} images were skipped.')
        actual_fns = []
        for i, (data, header) in enumerate(contents):
            if rms_arr[i] > rms_threshold:
                continue
            out_data += data / n
            actual_fns.append(fits_list[i])
        out_header = fits.Header()
        out_header['SRCFILES'] = ' '.join(fn.strip('/lustre/celery/') for fn in actual_fns)
        for k in contents[0][1]:
            if ('COMMENT' in k) or ('HISTORY' in k):
                continue
            out_header[k] = contents[0][1][k]
        fitsutils.write_image_fits(out_fn, out_header, out_data, overwrite=True)
    return out_fn
