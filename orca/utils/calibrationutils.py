import numpy as np
from os import path
import math
from astropy.time import Time
from astropy.coordinates import SkyCoord, ICRS
from orca.utils.constants import LWA_LON
from orca.utils.coordutils import CYG_A, OVRO_LWA_LOCATION, is_visible, get_altaz_at_ovro
from orca.utils.beam import beam
from casacore import tables
from casacore import measures
from casatools import componentlist


def BCAL_dadaname_list(utc_times_txt_path: str, duration: float = 20):
    """
    Get list of .dada file names to use for calibration. Selects .dada files that
    span {duration} centered on transit of Cygnus A.
    :param utc_times_txt_path: Path to utc_times.txt file.
    :param duration: In minutes, amount of time used for calibration.
        Default is 20 minutes.
    :return: List of .dada file names to use for calibration.
    """
    # If the utc_times.txt file contains multiple transits of Cygnus A, will select the
    # first transit.
    # Get array of UTC vals from {utc_times_txt_path}.
    utctimes, dadafiles = np.genfromtxt(utc_times_txt_path, delimiter=' \t ', \
        dtype='str', unpack=True)
    # Convert {utctimes} to array of LSTs and MJDs
    lsttimes = Time(utctimes, scale='utc').sidereal_time('apparent', LWA_LON)
    mjdtimes = Time(utctimes, scale='utc').mjd
    # Select {duration} range of LSTs where CygA is closest to zenith
    CygA_HA        = CYG_A.ra.hourangle
    CygA_HA_start  = CygA_HA - duration/2./60.
    CygA_HA_stop   = CygA_HA + duration/2./60.
    first_24_hrs   = np.where( (mjdtimes - mjdtimes[0]) <= 1. )
    rel_starttimes = np.abs(lsttimes.value[first_24_hrs] - CygA_HA_start)
    rel_stoptimes  = np.abs(lsttimes.value[first_24_hrs] - CygA_HA_stop)
    cyga_start_ind = list(rel_starttimes).index(min(rel_starttimes))
    cyga_stop_ind  = list(rel_stoptimes).index(min(rel_stoptimes))
    #
    BCALdadafiles   = dadafiles[first_24_hrs[0][cyga_start_ind]:first_24_hrs[0][cyga_stop_ind]+1]
    return BCALdadafiles


cal_srcs = [{'label': 'CasA', 'flux': '16530', 'alpha': -0.72, 'ref_freq': 80.0,
     'position': 'J2000 23h23m24s +58d48m54s'},
        {'label': 'CygA', 'flux': '16300', 'alpha': -0.58, 'ref_freq': 80.0,
     'position': 'J2000 19h59m28.35663s +40d44m02.0970s'}]


def flux80_47(flux_hi, sp, output_freq, ref_freq):
    # given a flux at 80 MHz and a sp_index,
    # return the flux at MS center-frequency.
    return flux_hi * 10 ** (sp * math.log(output_freq/ref_freq, 10))


def gen_model_ms_stokes(ms: str, zest: bool = False):
    """
    Generate component lists for calibration / polarized peeling in CASA. Currently only
        includes Cas A & Cyg A.
    :param ms: The measurement set.
    :param zest: For supplying component lists for polarized peeling. Default is False.
    :return: Returns path to component list(s). If zest=True, will return a list of paths
        to single-source component lists.
    """
    t0    = tables.table(ms, ack=False).getcell('TIME', 0)
    me    = measures.measures()
    time  = me.epoch('UTC', '%fs' % t0)
    timeT = Time(time['m0']['value'], format='mjd', scale='utc')
    timeT.format = 'datetime'
    utctime = timeT.value
    freq = float(tables.table(ms+'/SPECTRAL_WINDOW', ack=False).getcell('NAME', 0))/1.e6
    #
    outbeam = beam(ms)
    
    for s,src in enumerate(cal_srcs):
        ra  = src['position'].split(' ')[1]
        dec = src['position'].split(' ')[2]
        if is_visible(SkyCoord(ra, dec, frame=ICRS), utctime):
            altaz = get_altaz_at_ovro(SkyCoord(ra, dec, frame=ICRS), utctime)
            scale = np.array(outbeam.srcIQUV(altaz.az.rad, altaz.alt.rad))
            cal_srcs[s]['Stokes'] = str(list(flux80_47(float(src['flux']), src['alpha'], freq, src['ref_freq']) * scale))
        else:
            del cal_srcs[s]
    
    cl = componentlist()
    if zest:
        fluxIvals  = [float(src['Stokes'].replace('[','').replace(']','').split(',')[0]) for src in cal_srcs]
        sortedinds = np.argsort(fluxIvals)[::-1]
        cllist     = []
        counter    = 0
        for ind in sortedinds:
            src     = cal_srcs[ind]
            cl.done()
            cl.addcomponent(flux=eval(src['Stokes']), polarization='Stokes', dir=src['position'], label=src['label'])
            cl.setstokesspectrum(which=0, type='spectral index', index=[src['alpha'], 0, 0, 0], reffreq='%sMHz' % freq)            
            cl.rename('%s_%d.cl' % (path.splitext(path.abspath(ms))[0],counter))
            cl.done()
            cllist.append('%s_%d.cl' % (path.splitext(path.abspath(ms))[0],counter))
            counter+=1
        return cllist
    else:
        cl.done()
        for s,src in enumerate(cal_srcs):
            cl.addcomponent(flux=eval(src['Stokes']), polarization='Stokes', dir=src['position'], label=src['label'])
            cl.setstokesspectrum(which=s, type='spectral index', index=[src['alpha'], 0, 0, 0], reffreq='%sMHz' % freq)
        cl.rename('%s.cl' % path.splitext(path.abspath(ms))[0])
        cl.done()
        return '%s.cl' % path.splitext(path.abspath(ms))[0]
    