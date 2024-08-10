from orca.celery import app
from casatasks import deconvolve
from casatasks import importfits
#from random import choice
#from string import ascii_letters
import shutil

@app.task
def convert_and_deconvolve(fitsimage, fitspsf, msroot, niter=0, cleanup=True):
    """ Take an image and a psf to be deconvolved.
    Run casatask deconvolve.
    cleanup will remove mask, psf, residual, and model images but keep restored image in ms format.
    """

    # make temporary name for ms format files
#    randsuffix = "".join([choice(ascii_letters) for _ in range(3)])

    importfits(fitsimage=fitsimage, imagename=f'{msroot}.residual')
    importfits(fitsimage=fitspsf, imagename=f'{msroot}.psf')
    deconvolve(imagename=msroot, niter=niter, restoration=True)
    if cleanup:
        shutil.rmtree(f'{msroot}.mask')
        shutil.rmtree(f'{msroot}.psf')
        shutil.rmtree(f'{msroot}.residual')
        shutil.rmtree(f'{msroot}.model')

    return msroot
