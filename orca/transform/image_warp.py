from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import wcs

from scipy.interpolate import RBFInterpolator, CloughTocher2DInterpolator

import matplotlib.pyplot as plt

import numpy as np
from orca.celery import app

from multiprocess import Pool

import logging
import os
import sys
from time import time

# TODO: remove relative imports
# from source_detection import identify_sources_bdsf
# from catalogs import reference_sources_nvss
from source_detection import identify_sources_bdsf
from catalogs import reference_sources_nvss


logger = logging.getLogger(__name__)
logging.basicConfig(format="%(module)s:%(levelname)s:%(lineno)d %(message)s")
logger.setLevel(logging.INFO)

WORKING_DIR = "working"
OUTPUT_DIR = "outputs"
IMAGE_SIZE = 4096 # assume 4096x4096 images (specific to LWA)

# use half of the CPU cores available
CPU_COUNT = max(1, os.cpu_count() // 2)


def crossmatch(sources: SkyCoord, ref_sources: SkyCoord) -> SkyCoord:
    idx, d2d, d3d = sources.match_to_catalog_sky(ref_sources)
    return ref_sources[idx]


def compute_offsets(dxmodel, dymodel):
    # compute each row separately
    def calc_row(r):
        # all indices with row r
        xy =  np.indices((1, IMAGE_SIZE)).squeeze().transpose()  
        xy[:, 0] = r
        row_offsets = np.stack((dxmodel(xy), dymodel(xy)), axis=-1)
        return row_offsets
    
    # Naive multiprocessing (computing each row separately):
    # Note: while this should be extremely parallelizable , something (likely the GIL)
    # is preventing us from achieving optimal performance. This seems to take about 3
    # minutes with multiprocessing (64 cores) and 4.5 minutes without. Thus, Amdahl's
    # law tells us that only about 25% of this task is parallelizable (though it
    # should be closer to 100%).
    def go():
        res = None
        with Pool(processes=CPU_COUNT) as p:
            try:
                res = p.map(calc_row, list(range(IMAGE_SIZE)))
            except:
                p.close()
                import traceback
                raise Exception("".join(traceback.format_exception(*sys.exc_info())))
        return res

    results = go()
    return np.concatenate(results)


def compute_interpolation(interp):
    def g(r):
        xy =  np.indices((1, IMAGE_SIZE)).squeeze().transpose()
        xy[:, 0] = r
        return interp(xy)
    
    # naive multiprocessing, see above
    def go():
        res = None
        with Pool(processes=CPU_COUNT) as p:
            try:
                res = p.map(g, list(range(IMAGE_SIZE)))
            except:
                p.close()
                import traceback
                raise Exception("".join(traceback.format_exception(*sys.exc_info())))
                
        return res

    results = go()
    interp_img = np.stack(results, axis=0)
    return interp_img


def plot_separations(seps_before, seps_after, output_file=None):
    plt.figure()
    plt.hist([s.arcmin for s in seps_after], bins=100, log=True, fc=(1, 0, 0, 0.7))
    plt.hist([s.arcmin for s in seps_before], bins=100, log=True, fc=(0, 0, 1, 0.7))
    plt.title("Separations before (blue) and after (red) applying dewarping")
    plt.xlabel("Separation (arcmin)")
    plt.ylabel("Frequency")
    if output_file is not None:
        plt.savefig(output_file, dpi=300)
    else:
        plt.show()


def plot_image(image_data, title="", output_file=None):
    plt.figure()
    plt.imshow(image_data, interpolation='nearest', origin='lower', vmin=-1, vmax=15)
    plt.title(title)
    if output_file is not None:
        plt.savefig(output_file, dpi=300)
    else:
        plt.show()
    
@app.task
def image_plane_correction(img,
                           smoothing=350,
                           neighbors=20,
                           plot=False,
                          ):
    # get data from fits image
    image = fits.open(img)
    image_data = image[0].data[0, 0, :, :]
    imwcs = wcs.WCS(image[0].header, naxis=2)

    if img is not None:
        if not os.path.exists(WORKING_DIR):
            os.makedirs(WORKING_DIR)
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

    # identify sources from the image using pybdsf
    start = time()
    sources = identify_sources_bdsf(img, imwcs, WORKING_DIR)
    logger.info(f"Done identifying sources in {time() - start} seconds")
    logger.info(f"Found {len(sources)} sources")

    # we are using the NVSS catalog for reference sources
    ref_sources, _ = reference_sources_nvss(min_flux=100)
    logger.info(f"Using {len(ref_sources)} reference sources")

    # cross-match the sources found in the image with the reference sources
    logger.info("Crossmatching sources and reference sources")
    matched_ref_sources = crossmatch(sources, ref_sources)
    seps_before = sources.separation(matched_ref_sources)
    logger.info(f"Before correction: median separation of {np.median(seps_before).arcmin} arcmin")

    # pixel coordinates of sources and their corresponding reference sources
    sources_xy = np.stack(wcs.utils.skycoord_to_pixel(sources, imwcs), axis=1)
    ref_xy = np.stack(wcs.utils.skycoord_to_pixel(matched_ref_sources, imwcs), axis=1)

    # offsets between each source and its matched reference
    diff = ref_xy - sources_xy

    # learn an RBF model on the X and Y offsets independently
    # TODO: experiment with different parameters
    logger.info("Computing RBF interpolation models")
    dxmodel = RBFInterpolator(sources_xy, diff[:, 0], kernel='linear', smoothing=smoothing, neighbors=neighbors)
    dymodel = RBFInterpolator(sources_xy, diff[:, 1], kernel='linear', smoothing=smoothing, neighbors=neighbors)

    # the interpolated x and y offsets for each pixel, in row-major order
    logger.info("Computing offsets at every pixel")
    start = time()
    offsets = compute_offsets(dxmodel, dymodel)  # IMAGE_SIZE^2 x 2
    logger.info(f"Done computing offsets in {time() - start} seconds")

    # add the offset to each image index in the original image to move the pixel to a new location
    logger.info("Computing interpolation model for warped pixels")
    start = time()
    image_indices = np.indices((IMAGE_SIZE, IMAGE_SIZE)).swapaxes(0, 2)[:, :, ::-1].reshape((IMAGE_SIZE * IMAGE_SIZE, 2))
    interp = CloughTocher2DInterpolator(image_indices - offsets, np.ravel(image_data))
    logger.info(f"Done computing interpolation model in {time() - start} seconds")

    # compute interpolated image after applying offsets to each pixel
    logger.info("Dewarping the original image")
    start = time()
    dewarped = compute_interpolation(interp)
    logger.info(f"Done dewarping in {time() - start} seconds")

    # write dewarped image to a fits file
    output_img = np.expand_dims(np.expand_dims(dewarped, 0), 0)
    img_dewarp = os.path.basename(img).replace('.fits', '.dewarp.fits')
    fits.writeto(os.path.join(WORKING_DIR, img_dewarp), output_img, header=image[0].header, overwrite=True)

    # re-compute sources in interpolated image
    start = time()
    new_sources = identify_sources_bdsf(os.path.join(WORKING_DIR, img_dewarp), imwcs, WORKING_DIR)
    logger.info(f"Done identifying new sources in {time() - start} seconds")

    # compute source/reference separations in dewarped image
    new_matches = crossmatch(new_sources, ref_sources)
    seps_after = new_sources.separation(new_matches)
    logger.info(f"After correction: median separation of {np.median(seps_after).arcmin} arcmin")

    if plot:
        plot_separations(seps_before, seps_after, output_file=f"{OUTPUT_DIR}/separations.png")
        plot_image(image_data, "Original image", output_file=f"{OUTPUT_DIR}/original.png")
        plot_image(dewarped, "Dewarped", output_file=f"{OUTPUT_DIR}/dewarped.png")

    # cleanup
    image.close()
    del dxmodel
    del dymodel
    del interp

    # the "score", higher is better
    print(np.median(seps_before).arcmin - np.median(seps_after).arcmin)

    return os.path.join(WORKING_DIR, img_dewarp)