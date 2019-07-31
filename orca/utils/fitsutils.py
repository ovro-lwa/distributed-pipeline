from astropy.io import fits
import numpy as np


def read_image_fits(fn):
    with fits.open(fn) as hdulist:
        image = hdulist[0].data[0,0]
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
