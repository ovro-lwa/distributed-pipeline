#!/usr/bin/env python
"""Median filter preprocessing for source finding.

This module applies a median filter to FITS images to remove extended emission
and isolate point sources. Designed for preprocessing images before running
the source finding algorithm.

Example:
    Command-line usage::

        $ python source_find.medianfilter.py image.fits

    Creates image_medfilt.fits with extended emission subtracted.
"""

import pyfits
import scipy.signal
import numpy as np
import os,argparse


def medianfilter(fitsfile):
    """Apply median filter to a FITS image to isolate point sources.

    Subtracts a median-filtered version of the image to remove extended
    emission, leaving only point source structure.

    Args:
        fitsfile: Path to the input FITS image file.

    Note:
        Uses a 21x21 pixel kernel. Writes output with '_medfilt' suffix.
    """
    hdulist = pyfits.open(fitsfile)
    image   = hdulist[0].data[0,0].T
    header  = hdulist[0].header

    image_medfilt = scipy.signal.medfilt2d(image,kernel_size=21)
    pointsrcimg   = image - image_medfilt
    newfits       = pyfits.PrimaryHDU(np.asarray([np.asarray([pointsrcimg.T])]), header=header)
    newfits.writeto(os.path.splitext(os.path.abspath(fitsfile))[0]+'_medfilt.fits')


def main():
    """Parse command-line arguments and run median filter."""
    parser = argparse.ArgumentParser(description="Remove extended emission from image with a median filter. For use with sourcefind.20180326.py \
                                                  to identify point sources in image.")
    parser.add_argument("fitsfile", type=str, help="Path-to-fitsfile/fitsfile.fits")
    args = parser.parse_args()

    medianfilter(args.fitsfile)

if __name__ == '__main__':
    main()
