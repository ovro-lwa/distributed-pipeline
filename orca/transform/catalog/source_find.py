"""Source finding code.
Copied from source_find.20180502.py and changed import and some statements on initial commit.

# Source finding algorithm for finding transients.
# Written to work with sequential difference images.
# MMA - September 27 2017
"""

import numpy as np
import argparse
import sys
import os
import re
import pkg_resources

from astropy.io import fits as pyfits
import scipy.cluster.hierarchy as hcluster
from orca.extra.coords import utc2jd
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord
from scipy.optimize import fmin_l_bfgs_b

from billiard.pool import Pool

from typing import Optional, Tuple

EPSILON = 1e-30

mask = np.load(pkg_resources.resource_filename('orca', 'resources/mask_4096.npy'))

def fits_mask(image):
    """
    Apply horizon mask (copied pre-existing function from
        /home/mmanders/imaging_scripts/fits_mask.py
    and using pre-existing mask file from
        /home/mmanders/imaging_scripts/mask_4096.npz.
    Assumes image dimensions are 4096!!!
    Returns image with masked (NaN) pixels.
    """
    np.place(image, mask, np.nan)
    return image


def blockshaped(image,nrows,ncols):
    """
    Function that takes in image and returns X number of matrices corresponding to X
    divisions of the image.
    Taken from https://stackoverflow.com/a/16873755/190597
    Return an array of shape (n, nrows, ncols) where
        n * nrows * ncols = image.size
    If image is a 2D array, the returned array looks like n sub-blocks with each sub-block
    preserving the "physical" layout of image.
    """
    h, w = image.shape
    return (image.reshape(h//nrows, nrows, -1, ncols)
                 .swapaxes(1,2)
                 .reshape(-1, nrows, ncols))


def unblockshaped(image, h, w):
    """
    Undo the effects of blockshaped and reform image matrix.
    Taken from https://stackoverflow.com/a/16873755/190597
    Return an array of shape (h, w) where
        h * w = image.size
    If image is of shape (n, nrows, ncols), n sub-blocks of shape (nrows, ncols), then 
    the returns array preserves the "physical" layout of the sub-blocks.
    """
    n, nrows, ncols = image.shape
    return (image.reshape(h//nrows, -1, nrows, ncols)
                 .swapaxes(1,2)
                 .reshape(h, w))


def gauss2d(center, A, x0, y0, fwhmx, fwhmy, thetadeg, offset):
    """
    2D Gaussian function.
    fwhmx = BMAJ in pixels
    fwhmy = BMIN in pixels
    thetadeg = clock-wise rotation in degrees?
    """
    # Avoid divided by 0
    x, y = center
    theta  = thetadeg * np.pi/180.
    sigmax = fwhmx/2.35482 + EPSILON
    sigmay = fwhmy/2.35482 + EPSILON
    a   = np.cos(theta)**2 / (2*sigmax**2) + np.sin(theta)**2 / (2*sigmay**2)
    b   = -np.sin(2*theta) / (4*sigmax**2) + np.sin(2*theta) / (4*sigmay**2)
    c   = np.sin(theta)**2 / (2*sigmax**2) + np.cos(theta)**2 / (2*sigmay**2)
    ans = offset + A * np.exp(-( a*(x-x0)**2 + 2*b*(x-x0)*(y-y0) + c*(y-y0)**2 ))
    return ans.ravel()


def rot_ellipse(x, y, x0, y0, fwhmx, fwhmy, thetadeg):
    """
    Ellipse function, centered at x0, y0.
    fwhmx = 2 * semi-major axis in pixels = BMAJ
    fwhmy = 2 * semi-minor axis in pixels = BMIN
    thetadeg = counterclock-wise rotation in degrees?
    """
    theta  = thetadeg * np.pi/180.
    sigmax = fwhmx/2.
    sigmay = fwhmy/2.
    ans = ((x-x0)*np.cos(theta) + (y-y0)*np.sin(theta))**2. / sigmax**2. \
        + (-(x-x0)*np.sin(theta) + (y-y0)*np.cos(theta))**2. / sigmay**2
    return ans


def gauss2d_minimization(params, *args):
    """
    Function to be used with scipy.optimize.fmin_l_bfgs_b to find best-fit
    2D Gaussian parameters.
    See https://stackoverflow.com/questions/8672005/correct-usage-of-fmin-l-bfgs-b-for-fitting-model-parameters/8672743#8672743
    and https://en.wikipedia.org/wiki/Limited-memory_BFGS#L-BFGS-B
    """
    x                                   = args[0]
    y                                   = args[1]
    A,x0,y0,fwhmx,fwhmy,thetadeg,offset = params
    y_model                             = gauss2d(x,A,x0,y0,fwhmx,fwhmy,thetadeg,offset)
    error                               = y - y_model
    return np.sum(error**2.)


def sourcefit(xguess, yguess, peakguess, bmajpix, bminpix, bpahdr, locvals, imagecell):
    """
    Function that takes in image matrix (of any size), and returns the parameters for best
    fit Gaussian, in the following order:
    Returns - pkflux, xpos_rel, ypos_rel, bmaj_pix, bmin_pix, bpa
    """
    import warnings
    warnings.filterwarnings('error')
    initialguess  = (peakguess,xguess,yguess,bmajpix,bminpix,bpahdr,0)
    param_bounds  = [ (None,None), (0,imagecell.shape[0]-1), (0,imagecell.shape[1]-1),
                      (0,None), (0,None), (0,360), (None,None) ]
    locvalsx      = np.unique(locvals[0])
    locvalsy      = np.unique(locvals[1])
    x, y           = np.meshgrid(locvalsx,locvalsy)
    image_data    = np.asarray([imagecell[pixind] for pixind in zip(x.ravel(),y.ravel())])

    if np.sum(np.isnan(image_data)) >= 1:
        maskvals    = np.isnan(image_data)
        maskedarray = np.ma.fix_invalid(image_data, mask=maskvals.astype(int), fill_value=0)
        image_data  = maskedarray.data
    
    popt,func_value,info_dict = fmin_l_bfgs_b(gauss2d_minimization, x0=initialguess,
                                              args=((x,y),image_data),
                                              bounds=param_bounds, approx_grad=True)
    if info_dict['warnflag'] == 0:
        return popt[0],popt[1],popt[2],popt[3],popt[4],popt[5]
    if info_dict['warnflag'] in [1,2]:
        print("Too many function evaluations or iterations, could not converge.")
        return None


def sourcefind(imagecell: np.array, bmajpix, bminpix, bpahdr, plotimagecell: bool = False):
    """
    Function that takes in image matrix (of any size), and returns list of identified 
    sources.
    """
    rmscell = np.nanstd(imagecell.ravel()) # compute (rough) noise level in cell
    # identify locations in cell >5*rms
    #loc = np.where(np.isfinite(imagecell) & (imagecell >= 5*rmscell))
    locold = np.where(np.isfinite(imagecell) & (np.abs(imagecell) >= 5*rmscell))

    if np.asarray(locold).size == 0:
        return [],[],[],[],[],[],[] # no sources detected in this cell

    # get slightly "better" rms level after NaN-ing out >5sigma peaks in cell
    imagecellnopeaks         = np.copy(imagecell)
    imagecellnopeaks[locold] = np.nan
    rmscellbetter            = np.nanstd(imagecellnopeaks.ravel())
    loc = np.where(np.isfinite(imagecell) & (np.abs(imagecell) >= 5*rmscellbetter))
    
    if np.asarray(loc).size == 0:
        loc=locold
    
    # distance metric which also incorporates flux
    #def custom_metric(point1,point2):
    #    xpos1,ypos1,flux1 = point1
    #    xpos2,ypos2,flux2 = point2
    #    
    #    dist = np.sqrt((xpos2-xpos1)**2. + (ypos2-ypos1)**2.)
    #    fluxfrac = 1/flux2 + 1/flux1
    #    return dist*fluxfrac
    
    # identify clustering in >5sigma points
    if np.shape(loc)[1] > 1:
        X = np.transpose(loc)
        Z = hcluster.linkage(X)
        if len(Z[:,2]) == 1:
            clusters = hcluster.fclusterdata(X,1,criterion='distance')
            numclust = np.max(clusters)
        else:
            #X = np.vstack([loc[0],loc[1],imagecell[loc]]).T
            #dist = 50
            #dist = 25
            distsep=14
            topdendstep = Z[-1,2] - Z[-2,2]
            if distsep < topdendstep:
                dist = 14
            else:
                dist = Z[-1,2]
            clusters = hcluster.fclusterdata(X,dist,criterion='distance')
            clusterstmp = hcluster.fclusterdata(X,1,criterion='distance')
            #Z = hcluster.linkage(X, method='average', metric=custom_metric)
            #clusters = hcluster.fclusterdata(Z,5,criterion='maxclust')
            numclust = np.max(clusters)
            numclusttmp = np.max(clusterstmp)
            # evaluate clustering and adjust based on the following "hand-picked" criteria
            if np.max(Z[:,2]) == 1:
                pass
            elif len(Z[:,2]) == 2:
                pass
            elif len(Z[:,2]) == 3:
                pass
            elif len(Z[:,2]) == numclust:
                pass
            else:
                if numclust == 1:
                    selectstep = Z[-1,2] - Z[-2,2]
                    prevselectstep = 0
                    nextselectstep = Z[-2,2] - Z[-3,2]
                else:
                    selectstep = Z[-(numclust-1),2] - Z[-numclust,2] # step distance where algorithm
                                                                     # drew the line to demarcate clusters
                    prevselectstep = Z[-(numclust-1)+1,2] - Z[-numclust+1,2] # step distance immediately
                                                                             # above in the dendrogram
                    nextselectstep = Z[-(numclust-1)-1,2] - Z[-numclust-1,2] # step distance immediately
                                                                             # below in the dendrogram
                #if (selectstep < 2*distsep) and (selectstep > distsep) and (prevselectstep < distsep) and (nextselectstep < distsep):
                if (selectstep < 2*distsep) and (prevselectstep < distsep):
                    # increase dist to the value of selectstep
                    # this is to accommodate cases like: bright source (Tau A) being broken into multiple sources b/c another 
                    # distinct source is present elsewhere in the image cell.
                    dist = 2*distsep ##2*selectstep #Z[-(numclust-2),2]
                    clusters = hcluster.fclusterdata(X,dist,criterion='distance')
                    numclust = np.max(clusters)
            
            if plotimagecell:
                import pylab
                sys.setrecursionlimit(10000)
                pylab.ion()
                pylab.figure(figsize=(8,16))
                pylab.subplot(211)
                pylab.imshow(imagecell.T,cmap='gray',vmin=-4,vmax=4,origin='lower')
                ax = pylab.gca()
                for num in range(1,numclust+1):
                    #xval,yval = np.mean(X[np.where(clusters == num)][:,0:2],axis=0)
                    xval,yval = np.mean(X[np.where(clusters == num)],axis=0)
                    ax.annotate('%d' % num, fontsize=20, xy=(xval,yval), xycoords='data', color='red')
                for num in range(1,numclusttmp+1):
                    xval,yval = np.mean(X[np.where(clusterstmp == num)],axis=0)
                    ax.annotate('%d' % num, fontsize=15, xy=(xval,yval), xycoords='data',
                                xytext=(xval+25,yval+25), textcoords='data',
                                arrowprops=dict(arrowstyle="->",linewidth=1,color='pink'),
                                color='pink')
                pylab.subplot(212)
                hcluster.dendrogram(Z, leaf_rotation=90., leaf_font_size=8, color_threshold=dist)
    else:
        clusters = np.array([1])
        numclust = 1
    # iterate through clusters
    pkflux   = []
    xpos_rel = []
    ypos_rel = []
    bmaj_pix = []
    bmin_pix = []
    bpa      = []

    for num in range(1,numclust+1):
        clustinds    = np.where(clusters == num)
        locvals      = loc[0][clustinds],loc[1][clustinds]
        peakguess    = np.nanmax(np.abs(imagecell[locvals]))
        peakguessind = np.argmax(np.abs(imagecell[locvals]))
        xguess       = locvals[0][peakguessind]
        yguess       = locvals[1][peakguessind]
        #print num
        #print locvals
        #print np.shape(locvals)
        locvalxmin = np.min(locvals[0])-10
        locvalxmax = np.max(locvals[0])+10
        locvalymin = np.min(locvals[1])-10
        locvalymax = np.max(locvals[1])+10
        if locvalxmin < 0:
            locvalxmin = 0
        if locvalxmax >= imagecell.shape[0]:
            locvalxmax = imagecell.shape[0]-1
        if locvalymin < 0:
            locvalymin = 0
        if locvalymax >= imagecell.shape[1]:
            locvalymax = imagecell.shape[1]-1
        locvalsnewxx,locvalsnewyy = np.meshgrid(range(locvalxmin,locvalxmax),range(locvalymin,locvalymax))
        popt = sourcefit(xguess,yguess, peakguess, bmajpix, bminpix, bpahdr,
                         [locvalsnewxx.ravel(), locvalsnewyy.ravel()], np.abs(imagecell))

        if not popt:    # aka sourcefit returns None b/c it was not able to fit source
            continue
        pkflux.append(popt[0])
        xpos_rel.append(popt[1])
        ypos_rel.append(popt[2])
        bmaj_pix.append(popt[3])
        bmin_pix.append(popt[4])
        bpa.append(popt[5])

    return pkflux, xpos_rel, ypos_rel, bmaj_pix, bmin_pix, bpa, np.zeros(len(pkflux))+rmscellbetter


def sourcefind_multithread(fitsfile: str, beam: Tuple[float, float, float], n_proc: int = 16,
                           plot_cell: Optional[int] = None, plot_sources: bool = False):
    """
    Generate sources dictionary for input fitsfile.

    Args:
        fitsfile: Path to fits file.
        beam: (bmaj_deg, bmin_deg, bpa_deg).
        n_proc: number of parellel source_find processes (not exceeding the number of cells).
        plot_cell: the cell to create diagnostic plots on.
        plot_sources: whether to make plot for detected sources.

    Returns:

    """
    with pyfits.open(fitsfile) as hdulist:
        image   = hdulist[0].data[0,0].T
        header  = hdulist[0].header
    wcs     = WCS(header)
    pixscale= header['CDELT2']
    dateobs = header['date-obs']
    if header['BMAJ'] == 0:
        bmaj    = beam[0]
        bmin    = beam[1]
        bpa     = beam[2]
    else:
        bmaj    = header['BMAJ']
        bmin    = header['BMIN']
        bpa     = header['BPA']
    bmajpix = bmaj/pixscale
    bminpix = bmin/pixscale
    bpahdr  = bpa
    jdobs   = utc2jd.utc2jd(*filter(None, re.split('[-:T]', dateobs)))
    
    # apply horizon mask -- copied pre-existing function from
    #   /home/mmanders/imaging_scripts/fits_mask.py
    # and using pre-existing mask file from
    #   /home/mmanders/imaging_scripts/mask_4096.npz
    image_mask = fits_mask(image)

    # divide image into cells
    cell_shape = (4, 4)
    ncells = cell_shape[0] * cell_shape[1]
    nrows = image.shape[0] // cell_shape[0]
    ncols = image.shape[1] // cell_shape[1]
    imagecells  = blockshaped(image_mask, nrows, ncols)
    # divide image indices into same cells, in order to easily recover proper source
    # location later on
    x = np.arange(0,image_mask.shape[0])
    y = np.arange(0,image_mask.shape[1])
    x, y = np.meshgrid(x,y)
    xcells = blockshaped(x, nrows, ncols)
    ycells = blockshaped(y, nrows, ncols)

    # tmp = sourcefind(imagecells[9])
    # pdb.set_trace()
    #for num in range(0,16):
    #    print num
    #    tmp = sourcefind(imagecells[num])
    #    print tmp
    #    pdb.set_trace()


    # parallelize the source finding
    pkflux, xpos_rel, ypos_rel, bmaj_pix, bmin_pix, bpas, rmscell = [], [], [], [], [], [], []
    if n_proc > 1:
        pool = Pool(n_proc)
        pkflux,xpos_rel,ypos_rel,bmaj_pix,bmin_pix,bpas,rmscell = \
                zip(*pool.starmap(sourcefind, ((ic, bmajpix, bminpix, bpahdr) for ic in imagecells)))
    else:
        for imagecell in imagecells:
            pkflux_cell, xpos_rel_cell, ypos_rel_cell, bmaj_pix_cell, bmin_pix_cell, bpa_cell, rmscell_cell = \
                sourcefind(imagecell, bmajpix, bminpix, bpahdr)
            pkflux.append(pkflux_cell)
            xpos_rel.append(xpos_rel_cell)
            ypos_rel.append(ypos_rel_cell)
            bmaj_pix.append(bmaj_pix_cell)
            bmin_pix.append(bmin_pix_cell)
            bpas.append(bpa_cell)
            rmscell.append(rmscell_cell)

    pkflux_abs  = []
    ra_abs      = []
    dec_abs     = []
    xpos_abs    = []
    ypos_abs    = []
    bmaj_abs    = []
    bmin_abs    = []
    bpa_abs     = []
    rmscell_abs = []
    for cellnum in range(0,ncells):
        if not pkflux[cellnum]: # aka pkflux[cellnum] = []
            continue    # no sources detected in this cell

        # xyinds = zip(np.rint(xpos_rel[cellnum]).astype(int), np.rint(ypos_rel[cellnum]).astype(int))
        xinds = np.rint(xpos_rel[cellnum]).astype(np.intp)
        yinds = np.rint(ypos_rel[cellnum]).astype(np.intp)
        xcellnum = xcells[cellnum]
        ycellnum = ycells[cellnum]
        xpos_abs_cellnum = ycellnum[xinds, yinds]
        # xpos_abs_cellnum = [ycellnum[xyind] for xyind in xyinds]
        # ypos_abs_cellnum = [xcellnum[xyind] for xyind in xyinds]
        ypos_abs_cellnum = xcellnum[xinds, yinds]
        skycoords = pixel_to_skycoord(xpos_abs_cellnum, ypos_abs_cellnum, wcs)
        ra_abs_cellnum = skycoords.ra.value
        dec_abs_cellnum = skycoords.dec.value

        pkflux_abs.extend(np.array(pkflux[cellnum]))
        ra_abs.extend(np.array(ra_abs_cellnum))
        dec_abs.extend(np.array(dec_abs_cellnum))
        xpos_abs.extend(np.array(xpos_abs_cellnum))
        ypos_abs.extend(np.array(ypos_abs_cellnum))
        bmaj_abs.extend(np.array(bmaj_pix[cellnum]) * pixscale)
        bmin_abs.extend(np.array(bmin_pix[cellnum]) * pixscale)
        bpa_abs.extend(np.array(bpas[cellnum]))
        rmscell_abs.extend(np.array(rmscell[cellnum]))

    # save to numpy file
    outfilename = os.path.splitext(os.path.abspath(fitsfile))[0] + '_sfind'
    np.savez(outfilename, xpos_abs=xpos_abs, ypos_abs=ypos_abs, ra_abs=ra_abs,
             dec_abs=dec_abs, pkflux_abs=pkflux_abs, bmaj_abs=bmaj_abs, bmin_abs=bmin_abs,
             bpa_abs=bpa_abs, dateobs=dateobs, jdobs=jdobs, rmscell_abs=rmscell_abs)
    
    if plot_sources:
        _plot_detected_sources(bmaj_abs, bmin_abs, bpa_abs, fitsfile, image_mask, outfilename, pixscale, pkflux_abs,
                               xpos_abs, ypos_abs)
    return outfilename + '.npz'

def _plot_detected_sources(bmaj_abs, bmin_abs, bpa_abs, fitsfile, image_mask, outfilename, pixscale, pkflux_abs,
                           xpos_abs, ypos_abs):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    import warnings
    warnings.simplefilter("ignore")
    plt.figure(figsize=(20, 20))
    plt.imshow(image_mask.T, cmap='gray', vmin=-4, vmax=4, origin='lower')
    plt.title(os.path.splitext(os.path.basename(fitsfile))[0])
    ax = plt.gca()
    for sind, pkflux in enumerate(pkflux_abs):
        ax.annotate('%d' % sind, fontsize=15, xy=(xpos_abs[sind], ypos_abs[sind]),
                    xycoords='data', xytext=(xpos_abs[sind] + 75, ypos_abs[sind] + 75),
                    textcoords='data', arrowprops=dict(arrowstyle="->", linewidth=1,
                                                       color='pink'), color='pink')
        ell = Ellipse((xpos_abs[sind], ypos_abs[sind]), width=bmaj_abs[sind] / pixscale,
                      height=bmin_abs[sind] / pixscale, angle=-bpa_abs[sind],
                      edgecolor='red', facecolor='none', linewidth=1)
        ax.add_patch(ell)
    plt.savefig(outfilename + '.png')
    from PIL import Image
    Image.open(outfilename + '.png').save(outfilename + '.jpg', 'JPEG')


def main():
    parser = argparse.ArgumentParser(description="Generate sources dictionary for fitsfile.")
    parser.add_argument("fitsfile", type=str, help="Path-to-fitsfile/fitsfile.fits")
    parser.add_argument("--plot", default=False, action='store_true', help="Produce \
                        jpeg of source clustering.")
    args = parser.parse_args()

    # TODO make this an input argument
    beam = (0.33, 0.21, 56.)
    sourcefind_multithread(args.fitsfile, beam, plot_sources=args.plot)


if __name__ == '__main__':
    main()
