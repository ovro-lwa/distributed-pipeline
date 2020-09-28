import numpy as np
import glob
import os,argparse
import pylab
import numpy.ma as ma
from scipy.stats import skew
from scipy.ndimage import filters
from matplotlib.backends.backend_pdf import PdfPages
import casacore.tables as tables
from orca.wrapper import dada2ms
import pdb


def concat_dada2ms(dadafile_dir: str, BCALdadafile: str, outputdir: str):
    """
    Concatenate spws using dada2ms into single ms file. For passing to flag_bad_ants.
    """
    for spwind, dadapathwithspw in enumerate(np.sort(glob.glob(f'{dadafile_dir}/??/{BCALdadafile}'))):
        msfileconcat = os.path.splitext(os.path.basename(dadapathwithspw))[0]+'.ms'
        if spwind == 0:
            dada2ms.dada2ms(dadapathwithspw, f'{outputdir}/{msfileconcat}')
        else:
            dada2ms.dada2ms(dadapathwithspw, f'{outputdir}/{msfileconcat}', addspw=True)
    return f'{outputdir}/{msfileconcat}'
                    

def flag_ants_from_postcal_autocorr(msfile: str, tavg: bool = False) -> str:
    """Generates a text file containing the bad antennas.
    DOES NOT ACTUALLY APPLY FLAGS. CURRENTLY SHOULD ONLY BE RUN ON SINGLE SPW MSs.
    
    Args:
        msfile
        tavg: If set to True, will time average before evaluating flags.
        
    Returns:
        Path to the text file with the list of antennas to flag.
    """
    t      = tables.table(msfile, readonly=True)
    tautos = t.query('ANTENNA1=ANTENNA2')
    t.close()
    # get CORRECTED_DATA
    autos_corrected = tautos.getcol('CORRECTED_DATA')
    autos_flags     = tautos.getcol('FLAG')
    autos_antnums   = tautos.getcol('ANTENNA1')
    # autos_corrected.shape = (Nants*Nints, Nchans, Ncorrs)
    Nants = np.unique(autos_antnums).shape[0]
    Nints = int(autos_antnums.shape[0]/Nants)
    Ncorrs = autos_corrected.shape[-1]
    # average over frequency, reorder
    autos_corrected_mask = ma.masked_array(autos_corrected, mask=autos_flags, 
                                           fill_value=np.nan)
    autos_tseries = np.nanmean(autos_corrected_mask, axis=1).reshape(Nints, Nants, Ncorrs).transpose(1,0,2)
    antnums_reorder = autos_antnums.reshape(Nints, Nants).transpose(1,0)
    # autos_tseries.shape = (Nants, Nints, Ncorrs)
    # if msfile has Nints>1, use time series; else just take median
    if autos_tseries.shape[1] == 1:
        arr_to_evaluate = autos_tseries[:,0,:]
    elif tavg:
    	arr_to_evaluate = np.nanmean(autos_tseries,axis=1)
    else:
        medant_tseries  = np.nanmedian(autos_tseries, axis=0)
        arr_to_evaluate = np.nanstd(autos_tseries/medant_tseries, axis=1)
    # separate out core and expansion antennas
    inds_core = list(range(0,56)) + list(range(64,120)) + list(range(128,184)) + list(range(192,238))
    inds_exp  = list(range(56,64)) + list(range(120,128)) + list(range(184,192)) + list(range(238,246))
    medval_core = np.nanmedian(arr_to_evaluate[inds_core,:], axis=0)
    medval_exp = np.nanmedian(arr_to_evaluate[inds_exp,:], axis=0)
    stdval_core = np.std(arr_to_evaluate[inds_core,:], axis=0)
    stdval_exp = np.std(arr_to_evaluate[inds_exp,:], axis=0)
    # find 5sigma outliers, exclude, and recalculate stdval
    newinds_core = np.asarray(inds_core)[np.where( (arr_to_evaluate[inds_core,0] < medval_core[0]+3*stdval_core[0]) | 
                         (arr_to_evaluate[inds_core,3] < medval_core[3]+3*stdval_core[3]) )]
    newinds_exp = np.asarray(inds_exp)[np.where( (arr_to_evaluate[inds_exp,0] < medval_exp[0]+3*stdval_exp[0]) | 
                         (arr_to_evaluate[inds_exp,3] < medval_exp[3]+3*stdval_exp[3]) )]
    # exclude and recalculate
    medval_core = np.nanmedian(arr_to_evaluate[newinds_core,:], axis=0)
    medval_exp = np.nanmedian(arr_to_evaluate[newinds_exp,:], axis=0)
    stdval_core = np.std(arr_to_evaluate[newinds_core,:], axis=0)
    stdval_exp = np.std(arr_to_evaluate[newinds_exp,:], axis=0)

    newflagscore = np.asarray(inds_core)[np.where( (arr_to_evaluate[inds_core,0] > medval_core[0]+4*np.nanmin(stdval_core)) | 
                         (arr_to_evaluate[inds_core,3] > medval_core[3]+4*np.nanmin(stdval_core)) )]
    newflagsexp = np.asarray(inds_exp)[np.where( (arr_to_evaluate[inds_exp,0] > medval_exp[0]+4*np.nanmin(stdval_exp)) | 
                         (arr_to_evaluate[inds_exp,3] > medval_exp[3]+4*np.nanmin(stdval_exp)) )]
    flagsall = np.sort(np.append(newflagscore,newflagsexp))
    if flagsall.size > 0:
        antflagfile = os.path.splitext(os.path.abspath(msfile))[0]+'.ants'
        if os.path.exists(antflagfile):
            existingflags = np.genfromtxt(antflagfile, delimiter=',', dtype=int)
            flagsall = np.append(flagsall, existingflags)
            flagsall = np.unique(flagsall)
        flagsallstr = [str(flag) for flag in flagsall]        	
        flagsallstr2 = ",".join(flagsallstr)
        with open(antflagfile,'w') as f:
            f.write(flagsallstr2)
        return antflagfile
    else:
    	return None


def flag_bad_ants(msfile: str) -> str:
    """Generates a text file containing the bad antennas.
    DOES NOT ACTUALLY APPLY FLAGS.

    Args:
        msfile: msfile to generate

    Returns:
        Path to the text file with list of antennas to flag.
    """
    t       = tables.table(msfile, readonly=True)
    tautos  = t.query('ANTENNA1=ANTENNA2')
    
    # iterate over antenna, 1-->256
    datacolxx = np.zeros((256,2398))
    datacolyy = np.copy(datacolxx)
    for antind,tauto in enumerate(tautos.iter("ANTENNA1")):
        for bandind,tband in enumerate(tauto):
            datacolxx[antind,bandind*109:(bandind+1)*109] = tband["DATA"][:,0]
            datacolyy[antind,bandind*109:(bandind+1)*109] = tband["DATA"][:,3]

    datacolxxamp = np.sqrt( np.real(datacolxx)**2. + np.imag(datacolxx)**2. )
    datacolyyamp = np.sqrt( np.real(datacolyy)**2. + np.imag(datacolyy)**2. )

    datacolxxampdb = 10*np.log10(datacolxxamp/1.e2)
    datacolyyampdb = 10*np.log10(datacolyyamp/1.e2)

    # median value for every antenna
    medamp_perantx = np.median(datacolxxampdb,axis=1)
    medamp_peranty = np.median(datacolyyampdb,axis=1)

    # get flags based on deviation from median amp
    xthresh_pos = np.median(medamp_perantx) + np.std(medamp_perantx)
    xthresh_neg = np.median(medamp_perantx) - 2*np.std(medamp_perantx)
    ythresh_pos = np.median(medamp_peranty) + np.std(medamp_peranty)
    ythresh_neg = np.median(medamp_peranty) - 2*np.std(medamp_peranty)
    flags = np.where( (medamp_perantx > xthresh_pos) | (medamp_perantx < xthresh_neg) |\
                      (medamp_peranty > ythresh_pos) | (medamp_peranty < ythresh_neg) )

    # use unflagged antennas to generate median spectrum
    flagmask = np.zeros((256,2398))
    flagmask[flags[0],:] = 1
    datacolxxampdb_mask = ma.masked_array(datacolxxampdb, mask=flagmask, fill_value=np.nan)
    datacolyyampdb_mask = ma.masked_array(datacolyyampdb, mask=flagmask, fill_value=np.nan)

    medamp_allantsx = np.median(datacolxxampdb_mask,axis=0)
    medamp_allantsy = np.median(datacolyyampdb_mask,axis=0)

    stdarrayx = np.array( [np.std(antarr/medamp_allantsx) for antarr in datacolxxampdb_mask] )
    stdarrayy = np.array( [np.std(antarr/medamp_allantsy) for antarr in datacolyyampdb_mask] )
    
    # this threshold was manually selected...should be changed to something better at some point
    flags2 = np.where( (stdarrayx > 0.02) | (stdarrayy > 0.02) )

    flagsall = np.sort(np.append(flags,flags2))
    flagsallstr = [str(flag) for flag in flagsall]
    flagsallstr2 = ",".join(flagsallstr)

    antflagfile = os.path.dirname(os.path.abspath(msfile)) + '/flag_bad_ants.ants'
    with open(antflagfile,'w') as f:
        f.write(flagsallstr2)
    
    t.close()
    return antflagfile


def plot_autos(msfile: str):
    """
    Plot autocorrelations, grouped by ARX board.
    """
    # open MS tables
    t    = tables.table(msfile)
    tspw = tables.table(msfile+'/SPECTRAL_WINDOW')
    # initialize figure and pdf file
    pdffile = os.path.splitext(msfile)[0]+'_autos.pdf'
    pdf     = PdfPages(pdffile)
    pylab.figure(figsize=(15,10),edgecolor='Black')
    pylab.clf()
    ax1  = pylab.subplot(211)
    ax2  = pylab.subplot(212)
    ax1.set_prop_cycle(color=['blue','green','red','cyan','magenta','brown','black','orange'])
    ax2.set_prop_cycle(color=['blue','green','red','cyan','magenta','brown','black','orange'])
    legendstr = []
    # select autos from MS table
    tautos = t.query('ANTENNA1=ANTENNA2')
    # iterate over antennas
    for antind,tant in enumerate(tautos.iter("ANTENNA1")):
        ampXallbands = np.zeros(22*109)
        ampYallbands = np.copy(ampXallbands)
        freqallbands = np.copy(ampXallbands)
        tmpind = 0
        # iterate over subbands
        # (the tedious if statements are to correct for missing subbands in the ms file)
        for ind,tband in enumerate(tant):
            if ind != 0:
                if tspw[ind]['REF_FREQUENCY'] == tspw[ind-1]['REF_FREQUENCY']:
                    continue
                elif tspw[ind]['REF_FREQUENCY'] != (tspw[ind-1]['REF_FREQUENCY'] + tspw[ind-1]['TOTAL_BANDWIDTH']):
                    numpad = (tspw[ind]['REF_FREQUENCY'] - tspw[ind-1]['REF_FREQUENCY'])/tspw[ind-1]['TOTAL_BANDWIDTH'] - 1
                    amppad = np.zeros(109 * numpad) * np.nan
                    frqpadstart = tspw[ind-1]['REF_FREQUENCY'] + tspw[ind-1]['TOTAL_BANDWIDTH'] + tspw[ind-1]['EFFECTIVE_BW'][0]/2.
                    frqpadend   = frqpadstart + (tspw[ind-1]['TOTAL_BANDWIDTH'] * numpad)
                    frqpad = np.linspace(frqpadstart, frqpadend + tspw[ind-1]['EFFECTIVE_BW'][0]/2., tspw[ind-1]['NUM_CHAN']*numpad)
                    ampXallbands[tmpind*109:109*(tmpind+numpad)] = amppad
                    ampYallbands[tmpind*109:109*(tmpind+numpad)] = amppad
                    freqallbands[tmpind*109:109*(tmpind+numpad)] = frqpad
                    tmpind += numpad
            ampX = np.absolute(tband["DATA"][:,0])
            ampY = np.absolute(tband["DATA"][:,3])
            freq = tspw[ind]['CHAN_FREQ']
            ampXallbands[tmpind*109:109*(tmpind+1)] = ampX
            ampYallbands[tmpind*109:109*(tmpind+1)] = ampY
            freqallbands[tmpind*109:109*(tmpind+1)] = freq
            tmpind += 1
        legendstr.append('%03d' % (antind+1))
        ax1.plot(freqallbands/1.e6,10*np.log10(ampXallbands))
        ax2.plot(freqallbands/1.e6,10*np.log10(ampYallbands))
        # plot by ARX groupings
        if (np.mod(antind+1,8) == 0) and (antind != 0):
            pylab.xlabel('Frequency [MHz]')
            ax1.set_xticks(np.arange(0,100,2),minor=True)
            ax1.set_ylabel('Power [dB]')
            ax1.set_title('X',fontsize=18)
            ax1.set_ylim([40,100])
            ax2.set_xticks(np.arange(0,100,2),minor=True)
            pylab.ylabel('Power [dB]')
            ax2.set_title('Y',fontsize=18)
            ax2.set_ylim([40,100])
            ax1.legend(legendstr)
            ax2.legend(legendstr)
            if antind+1 in [64,128,192,248]:
                ax1.set_title('X -- fiber antennas',fontsize=18)
                ax2.set_title('Y -- fiber antennas',fontsize=18)
            elif antind+1 == 256:
                ax1.set_title('X -- leda antennas',fontsize=18)
                ax2.set_title('Y -- leda antennas',fontsize=18)
            elif antind+1 == 240:
                ax1.set_title('X -- fiber antennas 239,240',fontsize=18)
                ax2.set_title('Y -- fiber antennas 239,240',fontsize=18)
            pdf.savefig()
            # reiniatilize for new set of plots
            pylab.close()
            pylab.figure(figsize=(15,10),edgecolor='Black')
            ax1 = pylab.subplot(211)
            ax2 = pylab.subplot(212)
            ax1.set_prop_cycle(color=['blue','green','red','cyan','magenta','brown','black','orange'])
            ax2.set_prop_cycle(color=['blue','green','red','cyan','magenta','brown','black','orange'])
            legendstr = []
    pdf.close()
    return pdffile


def main():
    parser = argparse.ArgumentParser(description="Returns list of antennas to flag, based on power levels for \
                                                  autocorrelations in a single msfile. DOES NOT ACTUALLY FLAG \
                                                  THOSE ANTENNAS, JUST RETURNS A LIST TO BE FLAGGED.")
    parser.add_argument("msfile", help="Measurement set. Must be fullband measurement set, created with \
                                        ~/imaging_scripts/gen_autos.py.")
    args = parser.parse_args()
    flag_bad_ants(args.msfile)

if __name__ == '__main__':
    main()
