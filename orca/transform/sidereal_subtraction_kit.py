from orca.utils import fitsutils
import logging
import os
import numpy as np


def subtract_images(im1_path: str, im2_path: str, out_dir: str, psf_path: str=None, subtract_crab: bool=False,
                    shift: bool=False, scale: bool=False):
    logging.info(f'Subtracting {im2_path} by {im1_path}.')
    im1, header = fitsutils.read_image_fits(im1_path)
    im2, _ = fitsutils.read_image_fits(im2_path)
    if subtract_crab:
        #find the peak, subtract 0.85 of it; then find an adjacent peak, do like 0.4 percent
        logging.warn('The crab box is hardcoded.')
        psf, _ = fitsutils.read_image_fits(psf_path)
        im1 = sub_crab(im1, psf)
        im2 = sub_crab(im2, psf)
    if scale:
        # just minimize the stdev in the inner 1000 pixel by brute-forcing
        scales = np.linspace(0.90, 1.10, num=81)
        ind = np.argmin([np.std(
            x * im1[2048 - 500:2048 + 500, 2048 - 500:2048 + 500] - im2[2048 - 500:2048 + 500, 2048 - 500:2048 + 500])
            for x in scales])
        im1 = im1 * scales[ind]
    fitsutils.write_image_fits(f'{out_dir}/diff_{os.path.basename(im1_path)}', header, im2 - im1, overwrite=True)


def sub_crab(im, psf):
    im_T = im.T
    tauA = im_T[2190:2690, 1438:1808]
    peakx, peaky = np.unravel_index(np.argmax(tauA), tauA.shape)
    peakx += 2190
    peaky += 1438
    logging.info(f'peakx is {peakx} and peaky is {peaky}.')
    sub1 = sub_source(im_T, psf, 0.85, peakx, peaky)
    # Find the brightest pixel that's right next to the original peak and subtract
    displacements = [(-1, -1), (-1, 0), (-1, 1), (0, 0), (0, -1), (0, 1), (1, 1)]
    displacement = displacements[np.argmax([sub1[peakx + i, peaky + j] for i, j in displacements])]
    peakx += displacement[0]
    peaky += displacement[1]
    if sub1[peakx, peaky] > 50:
        sub2 = sub_source(sub1, psf, 0.5, peakx, peaky)
    else:
        logging.info(f'Not subtracting the second peak. with flux {sub1[peakx, peaky]}.')
        sub2=sub1
    return sub2.T


def sub_source(im, psf, gain, peakx, peaky):
    im_size = im.shape[0]
    psf_size = psf.shape[0]
    assert psf_size >= 2*im_size
    peakval = im[peakx, peaky]
    # Take a chunk centered at the peak, do the subtraction, write it back to the image
    trunc_psf = psf[psf_size//2 - peakx: psf_size//2 + im_size - peakx,
                   psf_size//2 - peaky : psf_size//2 + im_size - peaky]
    out_image = np.array(im)
    out_image -= gain * peakval * trunc_psf
    return out_image