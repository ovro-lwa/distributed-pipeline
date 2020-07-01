import numpy as np
import glob
import os,argparse
import pylab
import numpy.ma as ma
from scipy.stats import skew
from scipy.ndimage import filters
from matplotlib.backends.backend_pdf import PdfPages
import casacore.tables as tables
from orca.proj.boilerplate import run_dada2ms


def concat_dada2ms(dadafile_dir: str, BCALdadafile: str, outputdir: str):
    """
    Concatenate spws using dada2ms into single ms file. For passing to flag_bad_ants.
    """
    for spwind, dadapathwithspw in enumerate(np.sort(glob.glob(f'{dadafile_dir}/??/{BCALdadafile}'))):
        msfileconcat = os.path.splitext(os.path.basename(dadapathwithspw))[0]+'.ms'
        if spwind == 0:
            run_dada2ms(dadapathwithspw, f'{outputdir}/{msfileconcat}')
        else:
            run_dada2ms(dadapathwithspw, f'{outputdir}/{msfileconcat}', addspw=True)
    return f'{outputdir}/{msfileconcat}'
                    

def flag_bad_ants(msfile: str):
    """
    Input: msfile
    Returns list of antennas to be flagged based on autocorrelations.
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
