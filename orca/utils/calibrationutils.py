import numpy as np
from os import path
import math, shutil
from datetime import datetime
from astropy.time import Time
from astropy.coordinates import SkyCoord, ICRS
from orca.utils.coordutils import CYG_A, OVRO_LWA_LOCATION, is_visible, get_altaz_at_ovro
from casacore import tables
from casacore import measures
from casatools import componentlist

from orca.utils.beam import WoodyBeam as beam

import re
import os
import astropy.units as u

SRC_LIST = [{'label': 'CasA', 'flux': 16530, 'alpha': -0.72, 'ref_freq': 80.0,
             'position': 'J2000 23h23m24s +58d48m54s'},
            {'label': 'CygA', 'flux': 16300, 'alpha': -0.58, 'ref_freq': 80.0,
             'position': 'J2000 19h59m28.35663s +40d44m02.0970s'},
             {'label': 'VirA', 'flux': 2014, 'alpha': -0.79, 'ref_freq': 80.0,
                 'position': 'J2000 12h30m49.42338s +12d23m28.0439s'},
                {'label': 'TauA', 'flux': 1770, 'alpha': -0.27, 'ref_freq': 80.0,
                 'position': 'J2000 05h34m31.94s +22d00m52.2s'}]

def calibration_time_range(start_time: datetime, 
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
    start_lst = Time(start_time).sidereal_time('apparent', longitude=OVRO_LWA_LOCATION.lon)
    end_lst = Time(end_time).sidereal_time('apparent', longitude=OVRO_LWA_LOCATION.lon)
    # Select {duration_min} range of LSTs where CygA is closest to zenith
    CygA_HA        = CYG_A.ra.hourangle
    CygA_HA_start = (start_lst - CygA.ra).wrap_at(180*u.deg).hour
    CygA_HA_end = (end_lst - CygA.ra).wrap_at(180*u.deg).hour
    cal_start_time = start_time + timedelta(hours=(CygA_HA_start - duration_min/2./60.))
    cal_end_time = start_time + timedelta(hours=(CygA_HA_end + duration_min/2./60.))

    return cal_start_time, cal_end_time


def gen_model_ms_stokes(ms: str, zest: bool = False):
    """ Generate component lists for calibration / polarized peeling in CASA.

    Args:
        ms: Measurement set to generate model for.
        zest: For supplying component lists for polarized peeling. Default is False.

    Returns:
        Returns path to component list(s). If zest=True, will return a list of paths to 
        single-source component lists.
    """
    src_list = SRC_LIST
    with tables.table(ms, ack=False) as t:
        t0 = t.getcell('TIME', 0)
    me    = measures.measures()
    time  = me.epoch('UTC', '%fs' % t0)
    timeT = Time(time['m0']['value'], format='mjd', scale='utc')
    timeT.format = 'datetime'
    utctime = timeT.value
    lsttime = timeT.sidereal_time('apparent', OVRO_LWA_LOCATION.lon).value
    freq = float(tables.table(ms+'/SPECTRAL_WINDOW', ack=False).getcell('REF_FREQUENCY', 0))/1.e6
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

def gen_model_from_dict(ms: str, npzfile: str):
    """
    """
    Ateamdict = np.load(npzfile)
    sourcename_list = Ateamdict['sourcename']
    source_Vflux_list = Ateamdict['pkflux']
    source_ra_list = Ateamdict['ra_pos']
    source_dec_list = Ateamdict['dec_pos']
    freq_val = Ateamdict['frqval']
    src_list = [{'label': 'VirA', 'flux': '2400', 'alpha': -0.86, 'ref_freq': 80.0,
                 'position': 'J2000 12h30m49.42338s +12d23m28.0439s'},
                {'label': 'TauA', 'flux': '1770', 'alpha': -0.27, 'ref_freq': 80.0,
                 'position': 'J2000 05h34m31.94s +22d00m52.2s'},
                {'label': 'CasA', 'flux': 16530, 'alpha': -0.72, 'ref_freq': 80.0,
                 'position': 'J2000 23h23m24s +58d48m54s'}]
    freq = freq_val/1.e6
    sub_srcs = []
    for s,src in enumerate(src_list):
        if not np.isnan(source_Vflux_list[s]) and not np.isnan(source_ra_list[s]) and not np.isnan(source_dec_list[s]):
            src_list[s]['Stokes'] = [0, 0, 0, -source_Vflux_list[s]]
            new_pos = SkyCoord(source_ra_list[s], source_dec_list[s], frame='icrs', unit='deg')
            src_list[s]['position'] = f'J2000 {new_pos.to_string("hmsdms")}'
            sub_srcs.append(src_list[s])

    if not sub_srcs:
        return

    cl = componentlist()
    cl.done()
    clname = '%s.cl' % path.splitext(path.abspath(ms))[0]
    try:
        shutil.rmtree(clname)
    except OSError:
        pass
    for s,src in enumerate(sub_srcs):
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



def parse_filename(filename):
    pattern = r'(\d{8})_(\d{6})_.*\.ms'
    #match = re.match(pattern, filename)
    base_fname = os.path.basename(filename)
    match = re.match(pattern, base_fname)
    if not match:
        raise ValueError("Filename does not match the expected format 'YYYYMMDD_HHMMSS_*.ms'")
    date_str, time_str = match.groups()
    iso_time = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}T{time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}"
    return iso_time

def get_lst_from_filename(filename):
    utc_time_iso = parse_filename(filename)
    time = Time(utc_time_iso, format='isot', scale='utc')
    lst = time.sidereal_time('apparent', longitude=OVRO_LWA_LOCATION.lon)
    return lst

def source_ra_in_hours(position_str):
    parts = position_str.split()
    if len(parts) != 3:
        raise ValueError("Position string not in expected 'J2000 RA DEC' format.")
    ra_str, dec_str = parts[1], parts[2]
    coord = SkyCoord(ra_str, dec_str, frame=ICRS)
    return coord.ra.hour

def is_within_transit_window(filename, window_minutes=4):
    lst = get_lst_from_filename(filename).hour
    half_window_hours = (window_minutes / 2.) / 60.0
    in_window_sources = []
    for src in SRC_LIST:
        ra_h = source_ra_in_hours(src['position'])
        diff = abs((lst - ra_h + 12) % 24 - 12)
        if diff <= half_window_hours:
            in_window_sources.append(src)
    return in_window_sources

def get_relative_path(ms_path):
    # Extract the sub-path starting after '/slow/' or '/slow-averaged/'
    # For example, from '/lustre/pipeline/slow/73MHz/2024-11-29/00/20241129_000005_73MHz.ms'
    # we get '73MHz/2024-11-29/00/20241129_000005_73MHz.ms'
    if '/slow-averaged/' in ms_path:
        parts = ms_path.split('/slow-averaged/', 1)
        relative_path = parts[1].strip('/')
        return relative_path
    # If not found, fallback to '/slow/'
    elif '/slow/' in ms_path:
        parts = ms_path.split('/slow/', 1)
        relative_path = parts[1].strip('/')
        return relative_path
    else:
        raise ValueError("Input MS path does not contain '/slow/' or '/slow-averaged/'")


def build_output_paths(ms_path, base_output_dir='/lustre/pipeline/slow-averaged/'):
    relative_path = get_relative_path(ms_path)
    rel_dir = os.path.dirname(relative_path)  # e.g. '73MHz/2024-11-29/00'
    ms_base = os.path.splitext(os.path.basename(relative_path))[0]  # e.g. '20241129_000005_73MHz'
    output_dir = os.path.join(base_output_dir, rel_dir)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir, ms_base

