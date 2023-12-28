from orca.pipeline.boilerplate import run_dada2ms, flag_chans, apply_ant_flag, flag_ants, \
    apply_bl_flag, do_bl_flags, zest, run_chgcentre, run_integrate_with_concat, \
    do_calibration, get_spectrum, do_bandpass_correction, do_applycal, do_split, \
    run_wsclean
from orca.wrapper import change_phase_centre
from orca.utils.calibrationutils import calibration_time_range
from orca.utils.coordutils import CYG_A
from orca.flagging.flag_bad_ants import flag_bad_ants, concat_dada2ms, plot_autos, \
    flag_ants_from_postcal_autocorr

from orca.celery import app
from celery import group

from ..metadata.pathsmanagers import OfflinePathsManager

import logging, sys, os, shutil, glob
import numpy as np
from datetime import datetime, date


logging.basicConfig(stream=sys.stdout, level=logging.INFO)

user = os.getlogin()
# Point the PathsManager to location of data we will use for this tutorial
pm = OfflinePathsManager(
        utc_times_txt_path='/lustre/data/exoplanet/utc_times.txt.20210114',
        dadafile_dir='/lustre/data/exoplanet',
        working_dir=f'/lustre/{user}/tutorial',
        gaintable_dir=f'/lustre/{user}/tutorial/BCAL')
# Select the date-time range of data to process.
# For this tutorial, we will use 5 minutes of data, spanning 2019-12-06 14:00-14:05 UTC
start_time = datetime(2019,12,6,14,0,0)
end_time   = datetime(2019,12,6,14,5,0)

##########################
## CALIBRATION PIPELINE ##
##########################
def calibration_pipeline(start_time: datetime, end_time: datetime, pathman: OfflinePathsManager = pm):
    # identify calibration integrations
    cal_start_time, cal_end_time = calibration_time_range(pathman.utc_times_txt_path, 
                                                          start_time, end_time)
    pm_cal         = pathman.time_filter(start_time=cal_start_time, end_time=cal_end_time)
    BCALdadafiles  = [dadafile for dadafile in pm_cal.utc_times_mapping.values()]
    BCALdates      = [BCALdate for BCALdate in pm_cal.utc_times_mapping.keys()]
    middleBCALfile = BCALdadafiles[int(len(BCALdadafiles)/2)]
    middleBCALdate = BCALdates[int(len(BCALdates)/2)]
    # Raise error if calibration directory already exists and is not empty
    if os.path.exists(f'{pm_cal.gaintable_dir}/{middleBCALdate.date().isoformat()}'):
        raise FileExistsError(f'/lustre/{user}/tutorial/BCAL directory already exists. Will not overwrite.')

    # Get antenna flags and produce pdf of all antenna autocorrelations
    msfileantflags = concat_dada2ms(pm_cal.dadafile_dir, middleBCALfile, 
                        f'{pm_cal.gaintable_dir}/{middleBCALdate.date().isoformat()}')
    antflagfile    = flag_bad_ants(msfileantflags)
    ants           = np.genfromtxt(antflagfile, dtype=int, delimiter=',')
    pdffile        = plot_autos(msfileantflags)
    # chain together dada2ms and chgcentre commands
    phase_center = change_phase_centre.get_phase_center(msfileantflags)
    BCALmsfiles  = group([
                       run_dada2ms.s(pm_cal.get_dada_path(f'{s:02d}', t), 
        f'{pm_cal.gaintable_dir}/{middleBCALdate.date().isoformat()}/msfiles/' \
        f'{s:02d}_{t.isoformat()}.ms') | flag_chans.s(f'{s:02d}', crosshand=True, uvcut_m=30, usedatacol=True)
                       for t in pm_cal.utc_times_mapping.keys() for s in range(2,8)
                   ])()
    BCALmsfiles.get()
    #
    blfile1  = '/home/mmanders/imaging_scripts/flagfiles/defaults/expansion2expansion.bl'
    blfile2  = '/home/mmanders/imaging_scripts/flagfiles/defaults/flagsRyan_adjacent.bl'
    blfile3  = '/home/mmanders/imaging_scripts/flagfiles/defaults/withinARXboard.bl'
    blfile4  = '/home/mmanders/imaging_scripts/flagfiles/defaults/badbaselines.bl'
    ledafile = '/home/mmanders/imaging_scripts/flagfiles/defaults/leda.ants'
    ledaants = np.genfromtxt(ledafile, dtype=int, delimiter=',')
    result   = group([run_integrate_with_concat.s([
        f'{pm_cal.gaintable_dir}/{middleBCALdate.date().isoformat()}/msfiles/' \
        f'{s:02d}_{t.isoformat()}.ms' for t in pm_cal.utc_times_mapping.keys()], 
            pm_cal.get_gaintable_path(middleBCALdate.date(), f'{s:02d}', 'ms'),
            phase_center=phase_center) |
        apply_ant_flag.s(ants.tolist()) | apply_ant_flag.s(ledaants.tolist()) |
        apply_bl_flag.s(blfile1) | apply_bl_flag.s(blfile2) | apply_bl_flag.s(blfile3) | apply_bl_flag.s(blfile4) |
        do_calibration.s() | flag_ants.s() | flag_ants.s(tavg = True)
        for s in range(2,8)
    ])()
    msfiles = result.get()

    # move first pass cal tables to msfiles directory so they'll get deleted in the last step
    for s in range(2,8):
        caltableslist = [pm_cal.get_gaintable_path(middleBCALdate.date(), f'{s:02d}', calext) 
            for calext in ['bcal','X','dcal','cl']]
        for caltable in caltableslist:
            os.system(f'mv {caltable} {pm_cal.gaintable_dir}/{middleBCALdate.date().isoformat()}/msfiles')
    # do second pass calibration
    result2 = group([do_calibration.s(pm_cal.get_gaintable_path(middleBCALdate.date(), f'{s:02d}', 'ms')) |
        run_chgcentre.s(CYG_A.to_string('hmsdms')) |
        get_spectrum.s('CygA', timeavg=True) |
        do_bandpass_correction.s(pm_cal.get_bcal_path(middleBCALdate.date(),f'{s:02d}'),
                                 plot=True)
        for s in range(2,8)
    ])()
    bcal2files = result2.get()
    # delete single integration measurement sets
    os.system(f'rm -r {pm_cal.gaintable_dir}/{middleBCALdate.date().isoformat()}/msfiles')
    return middleBCALdate.date()

#########################
## PROCESSING PIPELINE ##
#########################
def processing_pipeline(CALdate: date, start_time: datetime, end_time: datetime, \
                        pathman: OfflinePathsManager = pm):
    pm      = pathman.time_filter(start_time=start_time, end_time=end_time)
    ants    = np.genfromtxt(
                  f'{pm.gaintable_dir}/{CALdate.isoformat()}/flag_bad_ants.ants', 
                  dtype=int, delimiter=',')
    blfile1 = '/home/mmanders/imaging_scripts/flagfiles/defaults/expansion2expansion.bl'
    blfile2 = '/home/mmanders/imaging_scripts/flagfiles/defaults/flagsRyan_adjacent.bl'
    blfile3 = '/home/mmanders/imaging_scripts/flagfiles/defaults/withinARXboard.bl'
    blfile4  = '/home/mmanders/imaging_scripts/flagfiles/defaults/badbaselines.bl'
    ledafile = '/home/mmanders/imaging_scripts/flagfiles/defaults/leda.ants'
    ledaants = np.genfromtxt(ledafile, dtype=int, delimiter=',')
    result = group([
        run_dada2ms.s(pm.get_dada_path(f'{s:02d}', t), out_ms=pm.get_ms_path(t, f'{s:02d}')) | 
        do_applycal.s([ pm.get_gaintable_path(CALdate, f'{s:02d}', calext) for calext in ['bcal','X','dcal','bcal2'] ]) | 
        apply_ant_flag.s(ants.tolist()) | apply_ant_flag.s(ledaants.tolist()) | flag_ants.s() |
        apply_bl_flag.s(blfile1) | apply_bl_flag.s(blfile2) | apply_bl_flag.s(blfile3) | apply_bl_flag.s(blfile4) |
        do_bl_flags.s() | flag_chans.s(f'{s:02d}', crosshand=True, uvcut_m=30) | do_bl_flags.s() |
        zest.s() | do_split.s()
        for t in pm.utc_times_mapping.keys() for s in range(2,8)
    ])()
    output = result.get()