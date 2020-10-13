#!/usr/bin/env python

import pyfits
import scipy.signal
import numpy as np
import os,argparse

def medianfilter(fitsfile):
    hdulist = pyfits.open(fitsfile)
    image   = hdulist[0].data[0,0].T
    header  = hdulist[0].header

    image_medfilt = scipy.signal.medfilt2d(image,kernel_size=21)
    pointsrcimg   = image - image_medfilt
    newfits       = pyfits.PrimaryHDU(np.asarray([np.asarray([pointsrcimg.T])]), header=header)
    newfits.writeto(os.path.splitext(os.path.abspath(fitsfile))[0]+'_medfilt.fits')


def main():
    parser = argparse.ArgumentParser(description="Remove extended emission from image with a median filter. For use with sourcefind.20180326.py \
                                                  to identify point sources in image.")
    parser.add_argument("fitsfile", type=str, help="Path-to-fitsfile/fitsfile.fits")
    args = parser.parse_args()

    medianfilter(args.fitsfile)

if __name__ == '__main__':
    main()
