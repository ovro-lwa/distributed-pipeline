"""
By Marin Anderson
"""
from astropy.io import fits
import os
from glob import glob
import numpy as np

# equation for a rotated ellipse centered at x0,y0
def rot_ellipse(x,y,x0,y0,sigmax,sigmay,theta):
    ans = ((x-x0)*np.cos(theta) + (y-y0)*np.sin(theta))**2. / sigmax**2. \
            + ((x-x0)*np.sin(theta) - (y-y0)*np.cos(theta))**2. / sigmay**2.
    return ans


def image_sub(file1,file2, out_dir, radius=0):
    hdulist1 = fits.open(file1)
    hdulist2 = fits.open(file2)
    # get image data and header information; Transpose such that the indices are consistent with ds9 and casa
    image1 = hdulist1[0].data[0,0].T
    image2 = hdulist2[0].data[0,0].T
    header1 = hdulist1[0].header
    naxis = header1['NAXIS1']
    freqval = header1['CRVAL3']
    dateobs = header1['DATE-OBS']
    hdulist1.close()
    hdulist2.close()

    # subtract images
    diffim = image2 - image1

    # write to file
    difffits = fits.PrimaryHDU(np.reshape(diffim.T, newshape=(1,1, *diffim.T.shape)), header=header1)
    difffits.writeto(out_dir + '/' + 'diff_%s' % os.path.basename(file1), overwrite=True)
    
    if radius==0:
        # get rms in center 1000x1000 pixels
        rmsval = np.std(diffim[naxis//2-500:naxis//2+500, naxis//2-500:naxis//2+500])
        medval = np.median(diffim[naxis//2-500:naxis//2+500, naxis//2-500:naxis//2+500])
        return rmsval, medval, freqval, dateobs
    elif radius > 0:
        xax = np.arange(0,naxis)
        x,y = np.meshgrid(xax,xax)
        apertureind = np.where(rot_ellipse(x.ravel(), y.ravel(), naxis//2., naxis//2., radius, radius, 0) <= 1.)
        imgind = zip(x.ravel()[apertureind], y.ravel()[apertureind])
        rmsval = np.std([diffim[subset] for subset in imgind])
        medval = np.median([diffim[subset] for subset in imgind])
        return rmsval, medval, freqval, dateobs


# sequentially subtracted images for 10day-run
# make sure to run in separate directory
def dir_sub(filepath,savediff=False,radius=0):
    filelist = np.sort(glob(filepath))
    rmsarray = np.zeros(len(filelist))
    medarray = np.copy(rmsarray)
    frqarray = np.copy(rmsarray)
    dateobsarray = np.copy(rmsarray)
    for ind in range(len(filelist)-1):
        print(ind)
        rmsval,medval,freqval,dateobs = image_sub(filelist[ind],filelist[ind+1],savediff=savediff,radius=radius)
        rmsarray[ind] = rmsval
        medarray[ind] = medval
        frqarray[ind] = freqval
        dateobsarray[ind] = dateobs
    return rmsarray,medarray,frqarray,dateobsarray


# sidereally subtracted images for 10day-run
# make sure to run in separate directory
def sid_sub(filepath):
    filelist = np.sort(glob(filepath))
    int_time = 30. # seconds
    ind_sep  = (3600 * 24. - 240.)/int_time
    for ind in range(len(filelist)):
        if ind+ind_sep <= len(filelist)-1:
            image_sub(filelist[ind], filelist[ind+ind_sep])
        else:
            image_sub(filelist[ind], filelist[ind-(np.floor(len(filelist)/ind_sep)-1)*ind_sep])
            

# take rms of image within given box
def getimrms(filepath,radius=0):
    filelist = np.sort(glob(filepath))
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
        imgind  = zip(x.ravel()[apertureind], y.ravel()[apertureind])
    for ind, file in enumerate(filelist):
        hdulist = fits.open(file)
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
