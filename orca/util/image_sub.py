"""
By Marin Anderson
"""
#!/usr/bin/env python
import pyfits
import os
from glob import glob
import numpy as np
import pylab as p
from scipy.ndimage import filters
from scipy.stats import nanstd
import pdb

# equation for a rotated ellipse centered at x0,y0
def rot_ellipse(x,y,x0,y0,sigmax,sigmay,theta):
    ans = ((x-x0)*np.cos(theta) + (y-y0)*np.sin(theta))**2. / sigmax**2. \
            + ((x-x0)*np.sin(theta) - (y-y0)*np.cos(theta))**2. / sigmay**2.
    return ans


def image_sub(file1,file2,savediff=False,radius=0):
    hdulist1 = pyfits.open(file1)
    hdulist2 = pyfits.open(file2)
    # get image data and header information
    image1   = hdulist1[0].data[0,0].T
    image2   = hdulist2[0].data[0,0].T
    header1  = hdulist1[0].header
    header2  = hdulist2[0].header
    naxis    = header1['NAXIS1']
    freqval  = header1['CRVAL3']
    dateobs  = header1['DATE-OBS']
    hdulist1.close()
    hdulist2.close()

    # subtract images
    diffim   = image2 - image1

    # write to file
    if savediff:
        difffits = pyfits.PrimaryHDU(np.asarray([np.asarray([diffim.T])]), header=header1)
        difffits.writeto('diff_%s' % os.path.basename(file1),clobber=True)
    
    if radius==0:
        # get rms in center 1000x1000 pixels
        rmsval = np.std(diffim[naxis/2.-500:naxis/2.+500,naxis/2.-500:naxis/2.+500])
        medval = np.median(diffim[naxis/2.-500:naxis/2.+500,naxis/2.-500:naxis/2.+500])
        return rmsval,medval,freqval,dateobs
    elif radius > 0:
        xax         = np.arange(0,naxis)
        x,y         = np.meshgrid(xax,xax)
        apertureind = np.where(rot_ellipse(x.ravel(),y.ravel(),naxis/2.,naxis/2.,radius,radius,0) <= 1.)
        imgind      = zip(x.ravel()[apertureind],y.ravel()[apertureind])
        rmsval      = np.std([diffim[subset] for subset in imgind])
        medval      = np.median([diffim[subset] for subset in imgind])
        return rmsval,medval,freqval,dateobs


# sequentially subtracted images for 10day-run
# make sure to run in separate directory
def dir_sub(filepath,savediff=False,radius=0):
    filelist = np.sort(glob(filepath))
    rmsarray = np.zeros(len(filelist))
    medarray = np.copy(rmsarray)
    frqarray = np.copy(rmsarray)
    dateobsarray = np.copy(rmsarray)
    for ind in range(len(filelist)-1):
        print ind
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
        hdulist = pyfits.open(filelist[0])
        header  = hdulist[0].header
        naxis   = header['NAXIS1']
        xax     = np.arange(0,naxis)
        x,y     = np.meshgrid(xax,xax)
        apertureind = np.where(rot_ellipse(x.ravel(),y.ravel(),naxis/2.,naxis/2.,radius,radius,0) <= 1.)
        imgind  = zip(x.ravel()[apertureind],y.ravel()[apertureind])
    for ind,file in enumerate(filelist):
        print ind
        hdulist = pyfits.open(file)
        header  = hdulist[0].header
        image   = hdulist[0].data[0,0].T
        naxis   = header['NAXIS1']
        freqval = header['CRVAL3']
        dateobs = header['DATE-OBS']
        if radius==0:
            # get rms in center 100x100 pixels
            rmsval = np.std(image[naxis/2.-500:naxis/2.+500,naxis/2.-500:naxis/2.+500])
            medval = np.median(image[naxis/2.-500:naxis/2.+500,naxis/2.-500:naxis/2.+500])
        else:
            rmsval      = np.std([image[subset] for subset in imgind])
            medval      = np.median([image[subset] for subset in imgind])
        rmsarray[ind] = rmsval
        medarray[ind] = medval
        frqarray[ind] = freqval
        dateobsarray.append(dateobs)

    return rmsarray,medarray,frqarray,np.array(dateobsarray)


# plot rms vals
def plotrms(rmsarrayf,medarrayf,frqarrayf,title=''):
    rmsarray = rmsarrayf[0:-1]
    medarray = medarrayf[0:-1]
    frqarray = frqarrayf[0:-1]
    # write arrays to textfile
    filename = title+'.txt'
    f = open(filename,'w')
    subbandarray = []
    channelarray = []
    for subband in np.arange(0,22):
        for channel in np.arange(0,109):
            subbandarray.append(subband)
            channelarray.append(channel)
    #f.write("# RMS [Jy] \t frequency [MHz] \t subband \t channel \t median pixel value \n")
    #for rms,med,frq,subband,channel in zip(rmsarray,medarray,frqarray/1.e6,subbandarray,channelarray):
        #f.write("%.02f \t %.03f \t %02d \t %03d \t %.02f \n" % (rms,frq,subband,channel,med))
    #f.close()
    # flag outlier values
    rmsarrayflag = np.copy(rmsarray)
    rmsarrayflag[np.where((rmsarray == 0.) | (rmsarray > 5*np.median(rmsarray)))] = np.nan
    # flag using median filter
    rmsarray_medfilt        = filters.median_filter(rmsarray,size=35)#20)
    stdmedfiltval           = nanstd(rmsarray/rmsarray_medfilt)
    rmsarray_medfiltflag    = np.where((rmsarray == 0) | (rmsarray > 3*stdmedfiltval+rmsarray_medfilt),np.nan,rmsarray)
    # PLOT RMS AND FLAGS
    channel_flags           = np.where(np.isnan(rmsarrayflag))[0]
    channel_flagsmedfilt    = np.where(np.isnan(rmsarray_medfiltflag))[0]
    p.figure(figsize=(15,10),edgecolor='Black')
    p.subplot(211)
    p.plot(frqarray/1.e6,rmsarray/rmsarray_medfilt,color='Black')
    #p.vlines((frqarray/1.e6)[channel_flagsmedfilt],0,100,color='Blue',linestyle='dashed')
    p.ylabel('Jy')
    p.title(title)
    p.ylim([0,100])
    p.xlim([20,90])
    p.subplot(212)
    p.plot(frqarray/1.e6,rmsarray_medfiltflag,color='Black')
    #p.plot(frqarray/1.e6,rmsarray_medfilt,color='Blue')
    #p.vlines((frqarray/1.e6)[channel_flagsmedfilt],0,100,color='Blue',linestyle='dashed')
    p.xlabel('Frequency [MHz]')
    p.ylabel('Jy')
    p.ylim([0,100])
    p.xlim([20,90])
    p.savefig('channelflags.png')
    p.show()

    # write out channel flags to file
    chfarray = []
    for ind,channelflag in enumerate(channel_flagsmedfilt):
        if (ind == 0) or (ind == len(channel_flagsmedfilt)-1) or ((channelflag+1 == channel_flagsmedfilt[ind+1]) and (channelflag-1 == channel_flagsmedfilt[ind-1])):
            chfarray.append("%02d:%03d" % (subbandarray[channelflag],channelarray[channelflag]))
            chfarray.append("%02d:%03d" % (subbandarray[channelflag+1],channelarray[channelflag+1]))
        elif (channelflag+1 == channel_flagsmedfilt[ind+1]) and (channelflag-1 != channel_flagsmedfilt[ind-1]):
            chfarray.append("%02d:%03d" % (subbandarray[channelflag+1],channelarray[channelflag+1]))
        elif (channelflag-1 == channel_flagsmedfilt[ind-1]) and (channelflag+1 != channel_flagsmedfilt[ind+1]):
            chfarray.append("%02d:%03d" % (subbandarray[channelflag],channelarray[channelflag]))
        else:
            chfarray.append("%02d:%03d" % (subbandarray[channelflag],channelarray[channelflag]))
            chfarray.append("%02d:%03d" % (subbandarray[channelflag+1],channelarray[channelflag+1]))
    chfarrayu = np.unique(chfarray)
    chf = open('channelflags.txt','w')
    for channelstr in chfarrayu:
        chf.write(channelstr+"\n")
    chf.close()

    # write out integration flags to file
    intarray = []
    for ind,intflag in enumerate(channel_flagsmedfilt):
        if ind == 0:
            continue
        if (ind == len(channel_flagsmedfilt)-1):
            intarray.append("%03d" % intflag)
        elif (intflag+1 == channel_flagsmedfilt[ind+1]) and (intflag-1 == channel_flagsmedfilt[ind-1]):
            intarray.append("%03d" % intflag)
            intarray.append("%03d" % (intflag+1))
        elif (intflag+1 == channel_flagsmedfilt[ind+1]) and (intflag-1 != channel_flagsmedfilt[ind-1]):
            intarray.append("%03d" % (intflag+1))
        elif (intflag-1 == channel_flagsmedfilt[ind-1]) and (intflag+1 != channel_flagsmedfilt[ind+1]):
            intarray.append("%03d" % intflag)
        else:
            intarray.append("%03d" % intflag)
            intarray.append("%03d" % (intflag+1))
    intarrayu = np.unique(intarray)
    intf = open('integrationflags.txt','w')
    for intstr in intarrayu:
        intf.write(intstr+"\n")
    intf.close()
    pdb.set_trace()
