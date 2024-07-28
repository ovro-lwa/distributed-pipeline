from astropy import wcs

from orca.utils import fitsutils

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
    data, header = fitsutils.read_image_fits(fits_file)
    w = wcs.WCS(header)
    return fitsutils.std_and_max_around_src(data.T, radius, coord, w)


