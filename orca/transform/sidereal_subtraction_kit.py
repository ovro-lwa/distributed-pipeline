from orca.utils import fitsutils
import logging
import os


def subtract_images(im1_path: str, im2_path: str, out_dir: str, psf_path: str=None, subtract_crab: bool=False,
                    shift: bool=False, scale: bool=False):
    logging.info(f'Subtracting {im2_path} by {im1_path}.')
    im1, header = fitsutils.read_image_fits(im1_path)
    im2, _ = fitsutils.read_image_fits(im2_path)
    if subtract_crab:
        #find the peak, subtract 0.85 of it; then find an adjacent peak, do like 0.4 percent
        logging.warn('The crab box is hardcoded.')
        pass
    if scale:
        # just minimize the stdev in the inner 500 pixel
        pass
    fitsutils.write_image_fits(f'{out_dir}/diff_{os.path.basename(im1_path)}', header, im2 - im1, overwrite=True)

