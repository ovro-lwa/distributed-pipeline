import numpy as np
import pylab
from os import path
from orca.proj.celery import app
from orca.utils.calibrationutils import gen_model_ms_stokes
from casatasks import clearcal, ft, bandpass, polcal, applycal
from casacore import tables


def calibration_steps(ms: str):
    """
    Perform basic calibration steps on measurement set, including generating model
    component list. Takes visibilities from the DATA column, and writes calibrated
    visibilities to CORRECTED_DATA.
    """
    # generate component list
    clfile = gen_model_ms_stokes(ms)
    # define calibration table names
    bcalfile = path.splitext(ms)[0]+'.bcal'
    Xcalfile = path.splitext(ms)[0]+'.X'
    dcalfile = path.splitext(ms)[0]+'.dcal'
    # CASA tasks
    clearcal(ms, addmodel=True)
    ft(ms, complist = clfile, usescratch=True)
    bandpass(ms, bcalfile, refant='34', uvrange='>15lambda', combine='scan,field,obs',
        fillgaps=1)
    polcal(ms, Xcalfile, gaintable=[bcalfile], refant='', uvrange='>15lambda>',
        poltype='Xf', combine='scan,field,obs')
    polcal(ms, dcalfile, gaintable=[bcalfile, Xcalfile], refant='', uvrange='>15lambda', 
        poltype='Dflls', combine='scan,field,obs')
    applycal(ms, gaintable=[bcalfile, Xcalfile, dcalfile], flagbackup=False)
    return ms
    

@app.task
def bandpass_correction(spectrumfile: str, bcalfile: str = None, plot: bool = False):
    """
    Generate calibration tables to correct bandpass flux scale based on Cyg A spectrum.
    :param spectrumfile: .npz file output by orca.transform.spectrum
    :param bcalfile: .bcal table to duplicate and fill with bandpass amplitude correction.
    :return: path to -spec.bcal file if {bcalfile} specified. Otherwise returns None.
    """
    spec = np.load(spectrumfile)
    # spec.files = {frqarr, timearr, speccorr, specI, specV}
    # model spectrum Baars et al.
    baarscyg = 10**(4.695 + 0.085 * np.log10(spec['frqarr']/1.e6) -0.178 * np.log10(spec['frqarr']/1.e6)**2.)
    #
    idx    = np.isfinite(spec['specI'])
    idnans = np.isnan(spec['specI'])
    #
    if plot:
        # initialize figure and pdf file
        pylab.figure(figsize=(25,8),edgecolor='Black')
        pylab.plot(spec['frqarr']/1.e6,spec['specI'],color='Black')
        pylab.plot(spec['frqarr']/1.e6,baarscyg,color='Blue')
        pylab.ylabel('Jy')
        pylab.xlabel('Frequency [MHz]')
        pylab.title('Cygnus A')
        pylab.xlim([spec['frqarr'].min(),spec['frqarr'].max()]/1.e6)
        pylab.ylim([15000,40000])
        pylab.savefig(path.splitext(spectrumfile)[0]+'_plot.png')
    #
    if bcalfile:
        # model / measured
        factor = 1/np.sqrt(baarscyg[idx] / spec['specI'][idx])
        # pad out "factor" with 1s where orig had nans
        factorfull         = np.zeros(len(spec['frqarr']))
        factorfull[idx]    = factor
        factorfull[idnans] = 1
        # open bcal file to populate CPARAM column
        tables.tablecopy(bcalfile, f'{path.splitext(bcalfile)[0]}-spec.bcal')
        t = tables.table(f'{path.splitext(bcalfile)[0]}-spec.bcal',readonly=False)
        # fill column
        gains    = t.getcol('CPARAM') # gains.shape = [Nants,Nchans,Npol]
        gains[:] = np.array([factorfull+0*1.j, factorfull+0*1.j], dtype=complex)
        t.putcol('CPARAM', gains)
        t.close()
        return f'{path.splitext(bcalfile)[0]}-spec.bcal'
    else:
        return None