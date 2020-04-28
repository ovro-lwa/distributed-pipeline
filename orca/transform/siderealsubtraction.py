from orca.utils import fitsutils
import logging
import os
import numpy as np

log = logging.getLogger(__name__)


def subtract_images(im1_path: str, im2_path: str, out_dir: str, psf_path: str=None, subtract_crab: bool = False,
                    scale: bool = False):
    logging.info(f'Subtracting {im2_path} by {im1_path}.')
    im1, header = fitsutils.read_image_fits(im1_path)
    im2, _ = fitsutils.read_image_fits(im2_path)
    if subtract_crab:
        log.warning('The crab box is hardcoded.')
        psf, _ = fitsutils.read_image_fits(psf_path)
        im1 = sub_crab(im1, psf)
        im2 = sub_crab(im2, psf)
    if scale:
        # just minimize the stdev in the inner 1000 pixel by brute-forcing
        scales = np.linspace(0.90, 1.10, num=101)
        ind = np.argmin(np.std(
            np.outer(scales, im1[2048 - 500:2048 + 500, 2048 - 500:2048 + 500].flatten()) -
            im2[2048 - 500:2048 + 500, 2048 - 500:2048 + 500].flatten(), axis=1))
        im1 = im1 * scales[ind]
    fitsutils.write_image_fits(f'{out_dir}/diff_{os.path.basename(im1_path)}', header, im2 - im1, overwrite=True)


def sub_crab(im, psf):
    im_T = im.T
    tauA = im_T[2190:2690, 1438:1808]
    init_peakx, init_peaky = np.unravel_index(np.argmax(tauA), tauA.shape)
    init_peakx += 2190
    init_peaky += 1438
    # Find the brightest pixel that's right next to the original peak and subtract
    square_around_crab = im_T[init_peakx-1:init_peakx+2, init_peaky-1: init_peaky+2]
    peakx, peaky = 1, 1
    sub = im_T
    while True:
        sub = sub_source(sub, psf.T, 0.4, peakx + init_peakx - 1, peaky + init_peaky - 1)
        peakx, peaky = np.unravel_index(np.argmax(square_around_crab), square_around_crab.shape)
        if square_around_crab[peakx, peaky] < 40:
            break
    return sub.T


def sub_source(im, psf, gain, peakx, peaky):
    im_size = im.shape[0]
    psf_size = psf.shape[0]
    assert psf_size >= 2*im_size
    peakval = im[peakx, peaky]
    # Take a chunk centered at the peak, do the subtraction
    trunc_psf = psf[psf_size//2 - peakx: psf_size//2 + im_size - peakx,
                   psf_size//2 - peaky : psf_size//2 + im_size - peaky]
    im -= gain * peakval * trunc_psf
    return im


