import numpy as np
from os import path
import math, shutil
from datetime import datetime
from astropy.time import Time
from astropy.coordinates import SkyCoord, ICRS
from orca.utils.coordutils import CYG_A, OVRO_LWA_LOCATION, is_visible, get_altaz_at_ovro
from orca.utils.beam import beam
from casacore import tables
from casacore import measures
from casatools import componentlist


def calibration_time_range(utc_times_txt_path: str, start_time: datetime, 
                           end_time: datetime, duration_min: float = 20):
    """Get dada file names based on Cygnus A transit for calibration.
    Get list of .dada file names to use for calibration. Selects .dada files that
    span {duration_min} centered on transit of Cygnus A.

    Args:
        utc_times_txt_path: Path to utc_times.txt file.
        start_time: Start time of data for which to derive calibration tables,
            datetime format.
        end_time: End time of data for which to derive calibration tables, 
            datetime format.
        duration_min: In minutes, amount of time used for calibration. Default is 20 min.

    Returns:
        cal_start_time and cal_end_time covering {duration_min} calibration range, 
            in datetime format.
    """
    # If the utc_times.txt file contains multiple transits of Cygnus A, will select the
    # first transit.
    # Get array of UTC vals from {utc_times_txt_path}.
    utctimes, dadafiles = np.genfromtxt(utc_times_txt_path, delimiter=' \t ', \
        dtype='str', unpack=True)
    # Convert {utctimes} to array of LSTs and MJDs
    lsttimes = Time(utctimes, scale='utc').sidereal_time('apparent',OVRO_LWA_LOCATION.lon)
    mjdtimes = Time(utctimes, scale='utc').mjd
    # Select {duration_min} range of LSTs where CygA is closest to zenith
    CygA_HA        = CYG_A.ra.hourangle
    CygA_HA_start  = CygA_HA - duration_min/2./60.
    CygA_HA_stop   = CygA_HA + duration_min/2./60.
    centermjd      = Time(datetime.fromtimestamp( 
                         (start_time.timestamp() + end_time.timestamp()) / 2. )).mjd
    rel_starttimes = np.abs(lsttimes.value - CygA_HA_start) + np.abs(mjdtimes - centermjd)
    rel_stoptimes  = np.abs(lsttimes.value - CygA_HA_stop) + np.abs(mjdtimes - centermjd)
    cyga_start_ind = list(rel_starttimes).index(min(rel_starttimes))
    cyga_stop_ind  = list(rel_stoptimes).index(min(rel_stoptimes))
    #
    cal_start_time = Time(utctimes[cyga_start_ind-1]).to_datetime()
    cal_end_time   = Time(utctimes[cyga_stop_ind+1]).to_datetime()
    #
    return cal_start_time, cal_end_time


def gen_model_ms_stokes(ms: str, zest: bool = False):
    """ Generate component lists for calibration / polarized peeling in CASA.
    Currently only includes Cas A & Cyg A.

    Args:
        ms: Measurement set to generate model for.
        zest: For supplying component lists for polarized peeling. Default is False.

    Returns:
        Returns path to component list(s). If zest=True, will return a list of paths to 
        single-source component lists.
    """
    src_list = [{'label': 'CasA', 'flux': 16530, 'alpha': -0.72, 'ref_freq': 80.0,
                 'position': 'J2000 23h23m24s +58d48m54s'},
                {'label': 'CygA', 'flux': 16300, 'alpha': -0.58, 'ref_freq': 80.0,
                 'position': 'J2000 19h59m28.35663s +40d44m02.0970s'}]    
    t0    = tables.table(ms, ack=False).getcell('TIME', 0)
    me    = measures.measures()
    time  = me.epoch('UTC', '%fs' % t0)
    timeT = Time(time['m0']['value'], format='mjd', scale='utc')
    timeT.format = 'datetime'
    utctime = timeT.value
    lsttime = timeT.sidereal_time('apparent', OVRO_LWA_LOCATION.lon).value
    freq = float(tables.table(ms+'/SPECTRAL_WINDOW', ack=False).getcell('NAME', 0))/1.e6
    #
    outbeam = beam(ms)

    cal_srcs = []
    for s,src in enumerate(src_list):
        src_position: str = src.get('position')    # type: ignore
        ra  = src_position.split(' ')[1]
        dec = src_position.split(' ')[2]
        if is_visible(SkyCoord(ra, dec, frame=ICRS), utctime) and \
        (not zest or (zest and (lsttime < 8 or lsttime > 13))):
            altaz = get_altaz_at_ovro(SkyCoord(ra, dec, frame=ICRS), utctime)
            scale = np.array(outbeam.srcIQUV(altaz.az.deg, altaz.alt.deg))
            src_list[s]['Stokes'] = list(_flux80_47(src['flux'], src['alpha'], freq, 
                                                    src['ref_freq']) * scale)
            cal_srcs.append(src_list[s])

    if not cal_srcs:
        return

    cl = componentlist()
    if zest:
        fluxIvals  = [src.get('Stokes')[0] for src in cal_srcs]    # type: ignore
        sortedinds = np.argsort(fluxIvals)[::-1]
        cllist     = []
        counter    = 0
        for ind in sortedinds:
            src     = cal_srcs[ind]
            clname  = '%s_%d.cl' % (path.splitext(path.abspath(ms))[0],counter)
            try:
                shutil.rmtree(clname)
            except OSError:
                pass
            cl.done()
            cl.addcomponent(flux=src['Stokes'], polarization='Stokes', 
                            dir=src['position'], label=src['label'])
            cl.setstokesspectrum(which=0, type='spectral index', 
                                 index=[src['alpha'], 0, 0, 0], reffreq='%sMHz' % freq)            
            cl.rename(clname)
            cl.done()
            cllist.append(clname)
            counter+=1
        return cllist
    else:
        cl.done()
        clname = '%s.cl' % path.splitext(path.abspath(ms))[0]
        try:
            shutil.rmtree(clname)
        except OSError:
            pass
        for s,src in enumerate(cal_srcs):
            cl.addcomponent(flux=src['Stokes'], polarization='Stokes', 
                            dir=src['position'], label=src['label'])
            cl.setstokesspectrum(which=s, type='spectral index', 
                                 index=[src['alpha'], 0, 0, 0], reffreq='%sMHz' % freq)
        cl.rename(clname)
        cl.done()
        return clname


def _flux80_47(flux_hi, sp, output_freq, ref_freq):
    # given a flux at 80 MHz and a sp_index,
    # return the flux at MS center-frequency.
    return flux_hi * 10 ** (sp * math.log(output_freq/ref_freq, 10))
