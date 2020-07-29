from orca.proj.boilerplate import run_dada2ms, flag_chans, apply_ant_flag, \
    apply_bl_flag, zest, run_chgcentre, run_integrate_with_concat, do_calibration, \
    get_spectrum, do_bandpass_correction, do_applycal
from orca.utils.calibrationutils import calibration_time_range
from orca.utils.coordutils import CYG_A
from orca.flagging.flag_bad_chans import flag_bad_chans
from orca.flagging.flag_bad_ants import flag_bad_ants, concat_dada2ms, plot_autos
from orca.wrapper import change_phase_centre
from orca.proj.celery import app
from celery import group
from ..metadata.pathsmanagers import OfflinePathsManager
import logging, sys, os
from os import path
import numpy as np
import glob
import pdb
from datetime import datetime, date
from casatasks import applycal

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

user = os.getlogin()
pm_20200117 = OfflinePathsManager(
                  utc_times_txt_path='/lustre/data/exoplanet_20200117/utc_times.txt',
                  dadafile_dir='/lustre/data/exoplanet_20200117',
                  working_dir=f'/lustre/{user}/exoplanet/orca_testwpm/LST_nopeel',
                  gaintable_dir=f'/lustre/{user}/exoplanet/orca_testwpm/LST_nopeel/BCAL')

start_time_testLSTnopeel = datetime(2020,1,22,9,30,0)
end_time_testLSTnopeel   = datetime(2020,1,22,11,30,0)

def calibration_pipeline_20200122():
    # identify calibration integrations
    cal_start_time, cal_end_time = calibration_time_range(pm_20200117.utc_times_txt_path, 
                                                          start_time_testLSTnopeel,
                                                          end_time_testLSTnopeel)
    pm_cal         = pm_20200117.time_filter(start_time=cal_start_time, 
                                             end_time=cal_end_time)
    BCALdadafiles  = [dadafile for dadafile in pm_cal.utc_times_mapping.values()]
    BCALdates      = [BCALdate for BCALdate in pm_cal.utc_times_mapping.keys()]
    middleBCALfile = BCALdadafiles[int(len(BCALdadafiles)/2)]
    middleBCALdate = BCALdates[int(len(BCALdates)/2)]
    # Get antenna flags and produce autocorrelation pdf
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
        f'{s:02d}_{t.isoformat()}.ms')
                       for t in pm_cal.utc_times_mapping.keys() for s in range(2,8)
                   ])()
    #
    blfile1 = '/home/mmanders/imaging_scripts/flagfiles/defaults/expansion2expansion.bl'
    blfile2 = '/home/mmanders/imaging_scripts/flagfiles/defaults/flagsRyan_adjacent.bl'
    group([run_integrate_with_concat.s([
        f'{pm_cal.gaintable_dir}/{middleBCALdate.date().isoformat()}/msfiles/' \
        f'{s:02d}_{t.isoformat()}.ms' for t in pm_cal.utc_times_mapping.keys()], 
            pm_cal.get_gaintable_path(middleBCALdate.date(), f'{s:02d}', 'ms'),
            phase_center=phase_center) |
        apply_ant_flag.s(ants.tolist()) |
        apply_bl_flag.s(blfile1) | apply_bl_flag.s(blfile2) |
        do_calibration.s() |
        run_chgcentre.s(CYG_A.to_string('hmsdms')) |
        get_spectrum.s('CygA', timeavg=True) |
        do_bandpass_correction.s(pm_cal.get_bcal_path(middleBCALdate.date(),f'{s:02d}'),
                                 plot=True) |
        run_chgcentre.si(pm_cal.get_gaintable_path(middleBCALdate.date(),f'{s:02d}','ms'),
                         phase_center) |
        do_applycal.s(
            [pm_cal.get_gaintable_path(middleBCALdate.date(), f'{s:02d}', calext) 
             for calext in ['bcal','X','dcal','bcal2']] ) |
        run_chgcentre.s(CYG_A.to_string('hmsdms')) |
        get_spectrum.s('CygA_postcorrection', timeavg=True) |
        do_bandpass_correction.s(plot=True)
        for s in range(2,8)
    ])()
    return middleBCALdate.date()


def processing_pipeline_testLSTnopeel(CALdate: date):
    pm      = pm_20200117.time_filter(start_time=start_time_testLSTnopeel,
                                      end_time=end_time_testLSTnopeel)
    ants    = np.genfromtxt(
                  f'{pm.gaintable_dir}/{CALdate.isoformat()}/flag_bad_ants.ants', 
                  dtype=int, delimiter=',')
    blfile1 = '/home/mmanders/imaging_scripts/flagfiles/defaults/expansion2expansion.bl'
    blfile2 = '/home/mmanders/imaging_scripts/flagfiles/defaults/flagsRyan_adjacent.bl'
    group([
        run_dada2ms.s(pm.get_dada_path(f'{s:02d}', t), 
                      out_ms=pm.get_ms_path(t, f'{s:02d}')) | 
        do_applycal.s([ pm.get_gaintable_path(CALdate, f'{s:02d}', calext) 
                        for calext in ['bcal','X','dcal','bcal2'] ]) | 
        apply_ant_flag.s(ants.tolist()) | 
        apply_bl_flag.s(blfile1) | 
        apply_bl_flag.s(blfile2) | 
        flag_chans.s(f'{s:02d}', crosshand=True, uvcut_m=50)
        for t in pm.utc_times_mapping.keys() for s in range(2,8)
    ])()

    
#@app.task
#def exoplanet_reverse_pipeline():