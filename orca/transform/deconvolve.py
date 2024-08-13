from orca.celery import app
from casatasks import deconvolve
from casatasks import importfits, exportfits
import shutil

@app.task
def convert_and_deconvolve(fitsimage, fitspsf, msroot, niter=0, cleanup=True):
    """ Take an image and a psf to be deconvolved with casatask deconvolve.
    Args:
        fitsimage: fits image to be deconvolved
        fitspsf: fits psf to be deconvolved
        msroot: root name for output ms format files
        niter: number of iterations for deconvolution
        cleanup: remove intermediate files

    Returns:
        path to deconvolved fits image
    """

    importfits(fitsimage=fitsimage, imagename=f'{msroot}.residual')
    importfits(fitsimage=fitspsf, imagename=f'{msroot}.psf')
    deconvolve(imagename=msroot, niter=niter, restoration=True)
    exportfits(imagename=f'{msroot}.image', fitsimage=f'{msroot}.image.fits')
    if cleanup:
        shutil.rmtree(f'{msroot}.mask')
        shutil.rmtree(f'{msroot}.psf')
        shutil.rmtree(f'{msroot}.residual')
        shutil.rmtree(f'{msroot}.model')
        shutil.rmtree(f'{msroot}.image')

    return f'{msroot}.image.fits'
