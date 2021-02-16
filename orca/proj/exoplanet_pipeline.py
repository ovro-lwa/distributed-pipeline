from orca.proj.boilerplate import run_dada2ms, flag_chans, apply_ant_flag, flag_ants, apply_bl_flag, do_bl_flags, zest, run_chgcentre, run_integrate_with_concat, do_calibration, get_spectrum, do_bandpass_correction, do_applycal, do_split, run_wsclean
from orca.utils.calibrationutils import calibration_time_range
from orca.utils.coordutils import CYG_A
from orca.flagging.flag_bad_ants import flag_bad_ants, concat_dada2ms, plot_autos, flag_ants_from_postcal_autocorr
from orca.wrapper import change_phase_centre
from orca.transform.image_sub import getimrms
from orca.utils.fitsutils import co_add
from orca.proj.celery import app
from celery import group
from ..metadata.pathsmanagers import OfflinePathsManager
import logging, sys, os, shutil
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
                  working_dir=f'/lustre/{user}/exoplanet/processing',
                  gaintable_dir=f'/lustre/{user}/exoplanet/processing/BCAL')
pm_20200219 = OfflinePathsManager(
                  utc_times_txt_path='/lustre/data/exoplanet_20200219/utc_times.txt',
                  dadafile_dir='/lustre/data/exoplanet_20200219',
                  working_dir=f'/lustre/{user}/exoplanet/processing',
                  gaintable_dir=f'/lustre/{user}/exoplanet/processing/BCAL')
pm_20191204 = OfflinePathsManager(
                  utc_times_txt_path='/lustre/data/exoplanet/utc_times.txt.20210114',
                  dadafile_dir='/lustre/data/exoplanet',
                  working_dir=f'/lustre/{user}/exoplanet/processing',
                  gaintable_dir=f'/lustre/{user}/exoplanet/processing/BCAL')
pm_20191121 = OfflinePathsManager(
                  utc_times_txt_path='/lustre/data/2019-11-21_rainydata/utc_times.txt',
                  dadafile_dir='/lustre/data/2019-11-21_rainydata',
                  working_dir=f'/lustre/{user}/exoplanet/processing',
                  gaintable_dir=f'/lustre/{user}/exoplanet/processing/BCAL')

start_time_testLSTnopeel = datetime(2020,1,22,9,30,0)
end_time_testLSTnopeel   = datetime(2020,1,22,11,30,0)
start_time_testLSTwpeel  = datetime(2020,1,22,18,30,30)
end_time_testLSTwpeel    = datetime(2020,1,22,20,30,0)
start_time_testLSTwpeel2 = datetime(2020,1,22,20,30,0)
end_time_testLSTwpeel2   = datetime(2020,1,22,21,30,0)

def calibration_pipeline(start_time: datetime, end_time: datetime, pathman: OfflinePathsManager = pm_20200117):
    # identify calibration integrations
    cal_start_time, cal_end_time = calibration_time_range(pathman.utc_times_txt_path, 
                                                          start_time, end_time)
    pm_cal         = pathman.time_filter(start_time=cal_start_time, 
                                             end_time=cal_end_time)
    BCALdadafiles  = [dadafile for dadafile in pm_cal.utc_times_mapping.values()]
    BCALdates      = [BCALdate for BCALdate in pm_cal.utc_times_mapping.keys()]
    middleBCALfile = BCALdadafiles[int(len(BCALdadafiles)/2)]
    middleBCALdate = BCALdates[int(len(BCALdates)/2)]
    ##############
    ## delete pre-existing BCAL directory!!!
    ##############
    #if path.exists(f'{pm_cal.gaintable_dir}/{middleBCALdate.date().isoformat()}'):
    #    os.system(f'rm -r {pm_cal.gaintable_dir}/{middleBCALdate.date().isoformat()}')
    # Get antenna flags and produce autocorrelation pdf
    msfileantflags = concat_dada2ms(pm_cal.dadafile_dir, middleBCALfile, 
                        f'{pm_cal.gaintable_dir}/{middleBCALdate.date().isoformat()}')
    antflagfile    = flag_bad_ants(msfileantflags)
    #antflagfile    = f'{pm_cal.gaintable_dir}/{middleBCALdate.date().isoformat()}/flag_bad_ants.ants'
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
#        run_chgcentre.s(CYG_A.to_string('hmsdms')) |
#        get_spectrum.s('CygA', timeavg=True) |
#        do_bandpass_correction.s(pm_cal.get_bcal_path(middleBCALdate.date(),f'{s:02d}'),
#                                 plot=True) |
#        run_chgcentre.si(pm_cal.get_gaintable_path(middleBCALdate.date(),f'{s:02d}','ms'),
#                         phase_center) |
#        do_applycal.s(
#            [pm_cal.get_gaintable_path(middleBCALdate.date(), f'{s:02d}', calext) 
#             for calext in ['bcal','X','dcal','bcal2']] ) |
#        run_chgcentre.s(CYG_A.to_string('hmsdms')) |
#        get_spectrum.s('CygA_postcorrection', timeavg=True) |
#        do_bandpass_correction.s(plot=True)
        for s in range(2,8)
    ])()
    msfiles = result.get()
#    for msfile in msfiles:
#        antflagfilebyspw = flag_ants_from_postcal_autocorr(msfile)
#        antflagfilebyspw2 = flag_ants_from_postcal_autocorr(msfile, tavg=True)
#        ants2 = np.genfromtxt(antflagfilebyspw, dtype=int, delimiter=',')
#        ants3 = np.genfromtxt(antflagfilebyspw2, dtype=int, delimiter=',')
#        apply_ant_flag(msfile,ants2.tolist())
#        apply_ant_flag(msfile,ants3.tolist())

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


def processing_pipeline(CALdate: date, start_time: datetime, end_time: datetime, pathman: OfflinePathsManager = pm_20200117):
    pm      = pathman.time_filter(start_time=start_time,
                                      end_time=end_time)
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
        run_dada2ms.s(pm.get_dada_path(f'{s:02d}', t), 
                      out_ms=pm.get_ms_path(t, f'{s:02d}')) | 
        do_applycal.s([ pm.get_gaintable_path(CALdate, f'{s:02d}', calext) 
                        for calext in ['bcal','X','dcal','bcal2'] ]) | 
        apply_ant_flag.s(ants.tolist()) | apply_ant_flag.s(ledaants.tolist()) |
        flag_ants.s() | 
        apply_bl_flag.s(blfile1) | apply_bl_flag.s(blfile2) | apply_bl_flag.s(blfile3) | apply_bl_flag.s(blfile4) |
        do_bl_flags.s() | flag_chans.s(f'{s:02d}', crosshand=True, uvcut_m=30) | do_bl_flags.s() | 
        zest.s()
        for t in pm.utc_times_mapping.keys() for s in range(2,8)
    ])()
    output = result.get()
    # concatenate in time per spw
    mslist = [pm.get_ms_path(t, '02') for t in pm.utc_times_mapping.keys()]
    middlems = mslist[int(np.floor(len(mslist)/2))]
    phase_center = change_phase_centre.get_phase_center(middlems)
    msfiles = group([run_integrate_with_concat.s([pm.get_ms_path(t, f'{s:02d}') 
        for t in pm.utc_times_mapping.keys()], f'{pm.working_dir}/twohourconcat/{s:02d}_{os.path.basename(middlems)[3::]}', phase_center=phase_center)
        for s in range(2,8)
    ])()
    msfiles_for_wsclean = msfiles.get()
    fitsfiles = run_wsclean(msfiles_for_wsclean, f'{pm.working_dir}/twohourconcat', \
    	f'{os.path.basename(middlems)[3::]}', \
        extra_arg_list=['-pol', 'I,V', '-j', '8', '-tempdir', '/dev/shm/mmanders', '-size', '4096', '4096', '-scale', '0.03125', '-weight', 'briggs', '0.5', '-taper-inner-tukey', '30'])


def processing_pipeline2(CALdate: date, start_time: datetime, end_time: datetime, pathman: OfflinePathsManager = pm_20200117):
    pm      = pathman.time_filter(start_time=start_time,
                                      end_time=end_time)
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
        run_dada2ms.s(pm.get_dada_path(f'{s:02d}', t), 
                      out_ms=pm.get_ms_path(t, f'{s:02d}')) | 
        do_applycal.s([ pm.get_gaintable_path(CALdate, f'{s:02d}', calext) 
                        for calext in ['bcal','X','dcal','bcal2'] ]) | 
        apply_ant_flag.s(ants.tolist()) | apply_ant_flag.s(ledaants.tolist()) |
        flag_ants.s() |
        apply_bl_flag.s(blfile1) | apply_bl_flag.s(blfile2) | apply_bl_flag.s(blfile3) | apply_bl_flag.s(blfile4) |
        do_bl_flags.s() | flag_chans.s(f'{s:02d}', crosshand=True, uvcut_m=30) | do_bl_flags.s() |
        zest.s() | do_split.s()
        for t in pm.utc_times_mapping.keys() for s in range(2,8)
    ])()
    output = result.get()

def imaging_pipeline(start_time: datetime, end_time:datetime, pathman: OfflinePathsManager = pm_20200117):
    pm = pathman.time_filter(start_time=start_time, end_time=end_time)
    # concatenate in time per spw
    mslist = [pm.get_ms_path(t, '02') for t in pm.utc_times_mapping.keys()]
    middlems = mslist[int(np.floor(len(mslist)/2))]
    phase_center = change_phase_centre.get_phase_center(middlems)
    msfiles = group([ run_chgcentre.s(pm.get_ms_path(t, f'{s:02d}'), phase_center) 
        for t in pm.utc_times_mapping.keys() for s in range(2,8)
    ])()
    snapshots = group([
        run_wsclean.s([pm.get_ms_path(t, f'{s:02d}') for s in range(2,8)], \
    	    pm.get_ms_parent_path(t), t.isoformat(), \
            extra_arg_list=['-pol', 'V', '-j', '1', '-tempdir', '/dev/shm/mmanders', '-size', '4096', '4096', '-scale', '0.03125', '-weight', 'briggs', '0.5', '-taper-inner-tukey', '30'])
        for t in pm.utc_times_mapping.keys()
    ])()
    fitsfile_prefixes = snapshots.get()
    fitsfiles = [f'{fitsfile_prefix}-dirty.fits' for fitsfile_prefix in fitsfile_prefixes]
    rmsarr, medarr, frqarr, timesarr = getimrms(fitsfiles, radius=5/0.03125)
    fitslist_forcoadd = np.asarray(fitsfiles)[np.where(rmsarr < np.median(rmsarr) + 2*np.std(rmsarr))]
    coaddfits = co_add(fitslist_forcoadd.tolist(), f'{pm.working_dir}/twohourconcat/{os.path.basename(middlems)[3::]}_coadd-V-dirty.fits')
    return timesarr[np.where(rmsarr >= np.median(rmsarr) + 2*np.std(rmsarr))]


def spectrum_pipeline(target_coordinates: str, target_name: str, start_time: datetime,
                      end_time: datetime, pathman: OfflinePathsManager = pm_20200117):
    pm      = pathman.time_filter(start_time=start_time, end_time=end_time)
    results = group([
        run_chgcentre.s(pm.get_ms_path(t, f'{s:02d}'), target_coordinates) |
        get_spectrum.s(target_name, 'DATA') 
        for t in pm.utc_times_mapping.keys() for s in range(2,8)
    ])()
    spectra = results.get()
    
    timestamps = [t.isoformat() for t in pm.utc_times_mapping.keys()]
    Nints = len(timestamps)
    spwcounter = 0
    intcounter = 0
    dynI = np.zeros((109*6,Nints))
    dynV = np.zeros((109*6,Nints))
    times = np.zeros(Nints)
    freqs = np.zeros(6*109)
    for spectrum in spectra:
        tmp = np.load(spectrum)
        dynI[109*spwcounter:109*(1+spwcounter),intcounter] = tmp['specI'][0]
        dynV[109*spwcounter:109*(1+spwcounter),intcounter] = tmp['specV'][0]
        times[intcounter] = tmp['timearr']
        if spwcounter == 5:
            spwcounter = 0
            intcounter += 1
        else:
            spwcounter += 1
    for ind,spectrum in enumerate(spectra[0:6]):
        tmp = np.load(spectrum)
        freqs[ind*109:109*(ind+1)] = tmp['frqarr']
    
    pdb.set_trace()    
    #import pylab as p
    #p.ion()
    #p.imshow(dynV, origin='lower', vmin=-100, vmax=100, extent=[0,553,freqs[0]/1.e6, 
    #         freqs[-1]/1.e6], aspect='auto', cmap='rainbow')
    #cbar = p.colorbar()
    #cbar.set_label('Jy', labelpad=20, fontsize=12)
    #p.xlabel('Integration number', fontsize=12)
    #p.ylabel('Frequency [MHz]', fontsize=12)
    #p.title(f'{target_name}, Stokes V', fontsize=14)