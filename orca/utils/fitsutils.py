from astropy.io import fits
import numpy as np
from typing import Tuple


def read_image_fits(fn: str) -> Tuple[np.array, fits.Header]:
    with fits.open(fn) as hdulist:
        image = hdulist[0].data[0, 0]
        header = hdulist[0].header
    return image, header


def image_to_np_index():
    """
    ds9/casa's images are the transpose of what astropy.io gives.
    This convers ds9 image pixel indices to the astropy.io numpy representation.
    :return:
    """
    pass


def write_image_fits(fn, header, data, overwrite=False):
    fits.PrimaryHDU(np.reshape(data, newshape=(1, 1, *data.shape)), header=header).writeto(
        fn, overwrite=overwrite)


def write_fits_mask_with_box(output_fits_path: str, imsize: int, center: Tuple[int, int], width: int) -> None:
    """
    Writes a fits mask to file.
    :param output_fits_path:
    :param imsize:
    :param center: Center of the box. Must be consistent with astropy indexing (i.e. the transpose of ds9 indexing)
    :param width:
    :return:
    """
    assert width % 2 == 1, 'width must be an odd number'
    image = np.zeros(shape=(imsize, imsize))
    image[(center[0] - width//2):(center[0] + width//2 + 1), (center[1] - width//2):(center[1] + width//2 + 1)] = \
        np.ones(shape=(width, width))
    write_image_fits(output_fits_path, get_sample_header(), image, overwrite=True)


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
