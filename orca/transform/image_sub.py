"""Image subtraction util by Marin Anderson
"""
from astropy.io import fits
import os
from glob import glob
import numpy as np

from orca.utils import fitsutils

from typing import Union, List

# equation for a rotated ellipse centered at x0,y0
def rot_ellipse(x,y,x0,y0,sigmax,sigmay,theta):
    ans = ((x-x0)*np.cos(theta) + (y-y0)*np.sin(theta))**2. / sigmax**2. \
            + ((x-x0)*np.sin(theta) - (y-y0)*np.cos(theta))**2. / sigmay**2.
    return ans


def image_sub(file1: Union[str, List[str]], file2: Union[str, List[str]], out_dir,
              out_prefix='diff_', ref_index: int = 0) -> str:
    """
    Image subtraction for single image for lists of images. When the input is a list, it co-adds first before
        subtracting. Subtraction is file2 - file1. Output file is based on the basename of file1 with a prefix.

    Args:
        file1: Previous image or list of images to be co-added
        file2: Next image or list of images to be co-added.
        out_dir: Output directory
        out_prefix: Output file prefix.
        ref_index: If the input is a list, the index for the reference fits file from which the header for the output
            is extracted.

    Returns:
        Output file path.
    """
    if isinstance(file1, list):
        assert isinstance(file1, list), 'Both input params must be the same type: list or single element.'
        im, header1 = fitsutils.read_image_fits(file1[ref_index])
        out_fn = os.path.basename(file1[ref_index])
        image1 = fitsutils.co_add_arr(file1, im.shape)
        image2 = fitsutils.co_add_arr(file2, im.shape)
    else:
        assert isinstance(file1, str)
        assert isinstance(file2, str)
        image1, header1 = fitsutils.read_image_fits(file1)
        image2, _ = fitsutils.read_image_fits(file2)
        out_fn = os.path.basename(file1)

    # subtract images
    diffim = image2 - image1

    # write to file
    out_path = f'{out_dir}/{out_prefix}{out_fn}'
    fitsutils.write_image_fits(out_path, header1, diffim, overwrite=True)
    return out_path


# take rms of image within given box
def getimrms(filepath: List[str], radius=0):
    #filelist = np.sort(glob(filepath))
    filelist = filepath
    rmsarray = np.zeros(len(filelist))
    medarray = np.zeros(len(filelist))
    frqarray = np.zeros(len(filelist))
    dateobsarray = [] 
    if radius != 0:
        hdulist = fits.open(filelist[0])
        header = hdulist[0].header
        naxis = header['NAXIS1']
        xax = np.arange(0, naxis)
        x, y = np.meshgrid(xax, xax)
        apertureind = np.where(rot_ellipse(x.ravel(), y.ravel(), naxis/2., naxis/2., radius, radius, 0) <= 1.)
        imgind  = list(zip(x.ravel()[apertureind], y.ravel()[apertureind]))
    for ind, filen in enumerate(filelist):
        hdulist = fits.open(filen)
        header = hdulist[0].header
        image = hdulist[0].data[0,0].T
        naxis = header['NAXIS1']
        freqval = header['CRVAL3']
        dateobs = header['DATE-OBS']
        if radius==0:
            rmsval = np.std(image[naxis//2-500:naxis//2+500, naxis//2-500:naxis//2+500])
            medval = np.median(image[naxis//2-500:naxis//2+500, naxis//2-500:naxis//2+500])
        else:
            rmsval = np.std([image[subset] for subset in imgind])
            medval = np.median([image[subset] for subset in imgind])
        rmsarray[ind] = rmsval
        medarray[ind] = medval
        frqarray[ind] = freqval
        dateobsarray.append(dateobs)

    return rmsarray,medarray,frqarray,np.array(dateobsarray)
