"""fits related utilities.
"""
from astropy.io import fits
from astropy import wcs
from astropy.coordinates import SkyCoord
import numpy as np
from typing import Tuple, List, Optional


def read_image_fits(fn: str) -> Tuple[np.ndarray, fits.Header]:
    with fits.open(fn) as hdulist:
        image = hdulist[0].data[0, 0]
        header = hdulist[0].header
    return image, header


def write_image_fits(fn, header, data, overwrite=False):
    fits.PrimaryHDU(np.reshape(data, newshape=(1, 1, *data.shape)), header=header).writeto(
        fn, overwrite=overwrite)


def write_fits_mask_with_box_xy_coordindates(output_fits_path: str, imsize: int,
                                             center_list: List[Tuple[int, int]], width_list: List[int]) -> str:
    """Writes a fits mask with a list of boxes to file.
    Writes a fits mask with a list of boxes to file. Both im+list and center_list are in
    WCS X Y coordindates (i.e. transpose of the numpy array
    from astropy.fits. The fits will NOT have a header that is accurate since only the pixels will be used.
    If you only need one box you can use one-element lists for the arguments.

    Args:
        output_fits_path:  The fits file path.
        imsize: size of the image.
        center_list: A list of enters of the boxes. Must be in XY index (i.e. what ds9 shows).
        width_list: A list of widths of the boxes in pixels, with each element being the width of the corresponding
    center_list element.

    Returns: The fits mask path.

    """
    image = np.zeros(shape=(imsize, imsize), dtype='>f4')
    for i, center in enumerate(center_list):
        width = width_list[i]
        assert width % 2 == 1, 'width must be an odd number'
        image[(center[0] - width//2):(center[0] + width//2 + 1), (center[1] - width//2):(center[1] + width//2 + 1)] = \
            np.ones(shape=(width, width))
        write_image_fits(output_fits_path, get_sample_header(), image.T, overwrite=True)
    return output_fits_path


def co_add(fits_list: List[str], output_fits_path: str, header_index: Optional[int] = None) -> str:
    im, header = read_image_fits(fits_list[header_index if header_index else len(fits_list) // 2])
    averaged_im = co_add_arr(fits_list, im.shape)
    write_image_fits(output_fits_path, header, averaged_im)
    return output_fits_path


def co_add_arr(fits_list, dims):
    averaged_im = np.zeros(shape=dims)
    n = len(fits_list)
    for fn in fits_list:
        im, _ = read_image_fits(fn)
        averaged_im += (im / n)
    return averaged_im


def get_peak_around_src(im_T: np.ndarray, source_coord: SkyCoord, w: wcs.WCS) -> Tuple[int, int]:
    x, y = wcs.utils.skycoord_to_pixel(source_coord, w)
    x_start = int(x) - 100
    y_start = int(y) - 100
    im_box = im_T[x_start:x_start + 200, y_start:y_start + 200]
    peakx, peaky = np.unravel_index(np.argmax(im_box),
                                    im_box.shape)
    peakx += x_start
    peaky += y_start
    return peakx, peaky


def std_and_max_around_src(im_T: np.ndarray, radius:int, source_coord: SkyCoord, w: wcs.WCS) -> Tuple[float, float]:
    x, y = wcs.utils.skycoord_to_pixel(source_coord, w)
    if np.isnan(x) or np.isnan(y):
        return np.nan, np.nan
    x = int(x)
    y = int(y)
    im_box = im_T[x - radius :x + radius, y - radius : y + radius]
    return np.std(im_box), np.max(im_box)


def get_sample_header() -> fits.Header:
    return fits.Header(
     {'SIMPLE': True,
      'BITPIX': -32,
      'NAXIS': 4,
      'NAXIS1': 4096,
      'NAXIS2': 4096,
      'NAXIS3': 1,
      'NAXIS4': 1,
      'EXTEND': True,
      'COMMENT':   "FITS (Flexible Image Transport System) format is defined in 'Astronomy and Astrophysics', "
                   "volume 376, page 359; bibcode: 2001A&A...376..359H",
      'BSCALE': 1.0,
      'BZERO': 0.0,
      'BUNIT': 'JY/BEAM',
      'BMAJ': 0.0,
      'BMIN': 0.0,
      'BPA': 0.0,
      'EQUINOX': 2000.0,
      'BTYPE': 'Intensity',
      'ORIGIN': 'WSClean',
      'CTYPE1': 'RA---SIN',
      'CRPIX1': 2049.0,
      'CRVAL1': 92.95164,
      'CDELT1': -0.03125,
      'CUNIT1': 'deg',
      'CTYPE2': 'DEC--SIN',
      'CRPIX2': 2049.0,
      'CRVAL2': 37.06079,
      'CDELT2': 0.03125,
      'CUNIT2': 'deg',
      'CTYPE3': 'FREQ',
      'CRPIX3': 1.0,
      'CRVAL3': 56148000.0,
      'CDELT3': 57552000.0,
      'CUNIT3': 'Hz',
      'CTYPE4': 'STOKES',
      'CRPIX4': 1.0,
      'CRVAL4': 1.0,
      'CDELT4': 1.0,
      'CUNIT4': '',
      'SPECSYS': 'TOPOCENT',
      'DATE-OBS': '2018-03-22T02:08:00.5',
      'WSCDATAC': 'CORRECTED_DATA',
      'WSCWEIGH': "Briggs'(0)",
      'WSCFIELD': 0.0,
      'WSCGAIN': 0.1,
      'WSCGKRNL': 7.0,
      'WSCIMGWG': 16184040.3775297,
      'WSCMGAIN': 1.0,
      'WSCNEGCM': 1.0,
      'WSCNEGST': 0.0,
      'WSCNITER': 0.0,
      'WSCNWLAY': 10.0,
      'WSCTHRES': 0.0})
