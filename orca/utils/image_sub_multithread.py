"""
By Marin Anderson
"""
#!/usr/bin/env python
import multiprocessing
import argparse,sys,os,re
import coords
import pdb

execfile('/lustre/mmanders/LWA/scripts/image_sub.py')

def image_sub_tmp(fitsfiles):
    rmsval,medval,frqval,dateobs = image_sub(fitsfiles[0],fitsfiles[1], savediff=True, radius=0)
    return rmsval,medval,frqval,dateobs

def image_rms(fitsfile):
    hdulist = pyfits.open(fitsfile)
    header  = hdulist[0].header
    diffim  = hdulist[0].data[0,0].T
    dateobs = header['DATE-OBS']
    naxis   = header['NAXIS1']
    freqval = header['CRVAL3']
    jdobs   = coords.utc2jd(*filter(None,re.split('[-:T]',dateobs)))
    # get rms in center 1000x1000 pixels
    rmsval = np.std(diffim[naxis/2.-500:naxis/2.+500,naxis/2.-500:naxis/2.+500])
    medval = np.median(diffim[naxis/2.-500:naxis/2.+500,naxis/2.-500:naxis/2.+500])
    return rmsval,medval,freqval,dateobs,jdobs

def image_dateobs(fitsfiles):
    hdulist_prev = pyfits.open(fitsfiles[0])
    hdulist      = pyfits.open(fitsfiles[1])
    header_prev  = hdulist_prev[0].header
    header       = hdulist[0].header
    dateobs_prev = header_prev['DATE-OBS']
    dateobs      = header['DATE-OBS']
    return dateobs_prev,dateobs

def image_sub_multithread(fitsfilesdir, startint, endint, poolnum=10):
    numarr              = np.arange(startint, endint+1)
    fitsfilearrint      = np.array( [fitsfilesdir+'/int%05d-dirty.fits' % num for num in numarr] )
    fitsfilearrint_prev = np.array( [fitsfilesdir+'/int%05d_prev-dirty.fits' % num for num in numarr] )

    pool = multiprocessing.Pool(poolnum)
    rmsarray,medarray,frqarray,dateobsarray = zip(*pool.map(image_sub_tmp, zip(fitsfilearrint_prev,fitsfilearrint)))
    jdobsarray = np.array( [coords.utc2jd(*filter(None,re.split('[-:T]',dateobs))) for dateobs in dateobsarray] )

    np.savez(fitsfilesdir+'/diffrms', rmsarray=rmsarray,medarray=medarray,frqarray=frqarray,dateobs=dateobsarray,jdobs=jdobsarray)

def get_rms_multithread(fitsfilesdir, poolnum=10):
    fitsfiles = np.sort(glob(fitsfilesdir+'/diff*.fits'))
    pool = multiprocessing.Pool(poolnum)
    rmsarray,medarray,frqarray,dateobsarray,jdobsarray = zip(*pool.map(image_rms, fitsfiles))

    np.savez(fitsfilesdir+'/diffrms_28hour',rmsarray=rmsarray,medarray=medarray,frqarray=frqarray,dateobsarray=dateobsarray,jdobsarray=jdobsarray)

def get_dateobs_multithread(fitsfilesdir, startint, endint, poolnum=10):
    numarr              = np.arange(startint, endint+1)
    fitsfilearrint      = np.array( [fitsfilesdir+'/int%05d-dirty.fits' % num for num in numarr] )
    fitsfilearrint_prev = np.array( [fitsfilesdir+'/int%05d_prev-dirty.fits' % num for num in numarr] )

    pool = multiprocessing.Pool(poolnum)
    dateobs_prevarray,dateobsarray = zip(*pool.map(image_dateobs, zip(fitsfilearrint_prev,fitsfilearrint)))

    np.savez(fitsfilesdir+'/dateobs', dateobs_prevarray=dateobs_prevarray, dateobsarray=dateobsarray)


def main():
    parser = argparse.ArgumentParser(description="Generate difference images, using multiprocessing.")
    parser.add_argument("fitsfilesdir", type=str, help="Fitsfile directory.")
    parser.add_argument("startint", type=int, help="Starting int number.")
    parser.add_argument("endint", type=int, help="Last int number.")
    parser.add_argument("--poolnum", dest='poolnum', type=int, default=10, action='store', help="Set number of threads for multithreading. Default is 10.")
    parser.add_argument("--justgetrms", default=False, action='store_true', help="Diff FITS already exist, just measure rms values.")
    parser.add_argument("--justgetdateobs", default=False, action='store_true', help="Get dateobs values from the headers of the int and int_prev .fits files.")

    args = parser.parse_args()

    if args.justgetrms:
        get_rms_multithread(args.fitsfilesdir, poolnum=args.poolnum)
    elif args.justgetdateobs:
        get_dateobs_multithread(args.fitsfilesdir, args.startint, args.endint, poolnum=args.poolnum)
    else:
        image_sub_multithread(args.fitsfilesdir, args.startint, args.endint, poolnum=args.poolnum)

if __name__ == '__main__':
    main()
