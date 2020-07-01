from orca.proj.boilerplate import run_dada2ms, flag_chans, apply_ant_flag, apply_bl_flag, zest, run_chgcentre, run_integrate_with_concat, do_calibration, get_spectrum
from orca.transform.calibration import bandpass_correction
from orca.utils.calibrationutils import BCAL_dadaname_list
from orca.utils.coordutils import CYG_A
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

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

#pm = OfflinePathsManager(utc_times_txt_path='/lustre/mmanders/exoplanet/orca_test/LST_wpeel/utc_times.txt',
#                         dadafile_dir='/lustre/data/exoplanet_20200117',
#                         msfile_dir='/lustre/mmanders/exoplanet/orca_test/LST_wpeel',
#                         bcal_dir='/lustre/mmanders/exoplanet/BCAL/20200123_multiintStokesV_take2')
user = os.getlogin()
utc_times_txt_path = '/lustre/mmanders/exoplanet/orca_test/LST_nopeel/utc_times.txt'
utc_times_txt_pathcal = '/lustre/mmanders/exoplanet/orca_test/LST_nopeel/utc_times_24hr.txt'
dadafile_dir       = '/lustre/data/exoplanet_20200117'
msfile_dir         = f'/lustre/{user}/exoplanet/orca_test/LST_nopeel'
os.makedirs(msfile_dir, exist_ok=True)
bcal_dir           = f'/lustre/{user}/exoplanet/orca_test/LST_nopeel/BCAL'
os.makedirs(bcal_dir, exist_ok=True)
flag_dir           = f'/home/{user}/imaging_scripts/flagfiles/20200122'
os.makedirs(flag_dir, exist_ok=True)
dadafilelist       = np.genfromtxt(utc_times_txt_path, delimiter=' \t ', unpack=True, dtype=str)[1]
spws               = [path.basename(dadafile_dir_wspw) for dadafile_dir_wspw in np.sort(glob.glob(f'{dadafile_dir}/??'))]


def calibration_pipeline(utc_times_txt_path: str):
    ## add in os.makedirs!
    # identify calibration integrations
    BCALdadafiles  = BCAL_dadaname_list(utc_times_txt_path)
    middleBCALfile = BCALdadafiles[int(BCALdadafiles.shape[0]/2)]
    # Get antenna flags and produce autocorrelation pdf
    msfileantflags = concat_dada2ms(dadafile_dir, middleBCALfile, bcal_dir)
    antflagfile    = flag_bad_ants(msfileantflags)
    pdffile        = plot_autos(msfileantflags)
    # chain together dada2ms and chgcentre commands
    phase_center = change_phase_centre.get_phase_center(msfileantflags)
    pdb.set_trace()
    for spw in spws:
        BCALmsfiles  = group( [run_dada2ms(f'{dadafile_dir}/{spw}/{dadafile}',f'{bcal_dir}/{spw}-{path.splitext(dadafile)[0]}.ms') for dadafile in BCALdadafiles] )()
        BCALmsfiles  = group( [run_chgcentre(msfile, phase_center) for msfile in BCALmsfiles] )()
    BCALmslist   = group( [run_integrate_with_concat(np.sort(glob.glob(f'{bcal_dir}/{spw}-*.ms')), f'{bcal_dir}/{spw}-T1al.ms') for spw in spws] )()
    # flag antennas, baselines
    #ants = np.genfromtxt(f'{flag_dir}/all.antflags', dtype=int, delimiter=',')
    ants         = np.genfromtxt(antflagfile, dtype=int, delimiter=',')
    BCALmslist   = group( [apply_ant_flag(msfile, ants) for msfile in BCALmslist] )()
    #blfile       = f'{flag_dir}/all.blflags'
    blfile1      = '/home/mmanders/imaging_scripts/flagfiles/defaults/expansion2expansion.bl'
    BCALmslist   = group( [apply_bl_flag(msfile, blfile1) for msfile in BCALmslist] )()
    blfile2      = '/home/mmanders/imaging_scripts/flagfiles/defaults/flagsRyan_adjacent.bl'
    BCALmslist   = group( [apply_bl_flag(msfile, blfile2) for msfile in BCALmslist] )()
    # Basic calibration steps
    BCALmslist   = group( [do_calibration(msfile) for msfile in BCALmslist] )()
    # chgcentre and generate spectrum
    BCALmslist   = group( [run_chgcentre(msfile, CYG_A.to_string('hmsdms')) for msfile in BCALmslist] )()
    spectrafiles = group( [get_spectrum(msfile, 'CygA', timeavg=True) for msfile in BCALmslist] )()
    # bandpass correction tables
    bpasscorrlist = group( [bandpass_correction(spectrumfile, path.splitext(msfile)[0]+'.bcal', plot=True) for spectrumfile,msfile in zip(spectrafiles,BCALmslist)] )()
    

@app.task
def processing_pipeline(dadafile = str, subband = int):
    spw    = '%02d' % subband
    # dada2ms
    msfile = run_dada2ms(f'{dadafile_dir}/{spw}/{dadafile}', 
                         f'{msfile_dir}/{spw}-{path.splitext(dadafile)[0]}.ms')
    # applycal
    bcal    = f'{bcal_dir}/{spw}-T1al.bcal'
    Xcal    = f'{bcal_dir}/{spw}-T1al.X'
    dcal    = f'{bcal_dir}/{spw}-T1al.dcal'
    #cygacal = bcal_dir+'/'+spw+'-T1al-spec.bcal'
    #applycal(msfile, gaintable=[bcal,Xcal,dcal,cygacal], flagbackup=False)
    applycal(msfile, gaintable=[bcal,Xcal,dcal], flagbackup=False)
    # apply antenna flags
    ants = np.genfromtxt(f'{flag_dir}/all.antflags', dtype=int, delimiter=',')
    apply_ant_flag(msfile, ants)
    # apply baseline flags
    blfile = f'{flag_dir}/all.blflags'
    apply_bl_flag(msfile,blfile)
    # generate and apply channel flags
    flag_chans(msfile, spw)
    # polarized peel
    zest(msfile)
    return msfile
    
#@app.task
#def exoplanet_reverse_pipeline():