from orca.proj.boilerplate import run_dada2ms, flag_chans, apply_ant_flag, apply_bl_flag, zest
from .celery import app
from celery import group
from ..metadata.pathsmanagers import OfflinePathsManager
from casatasks import applycal
import logging, sys
from os import path
import numpy as np
import glob

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

#pm = OfflinePathsManager(utc_times_txt_path='/lustre/mmanders/exoplanet/orca_test/LST_wpeel/utc_times.txt',
#                         dadafile_dir='/lustre/data/exoplanet_20200117',
#                         msfile_dir='/lustre/mmanders/exoplanet/orca_test/LST_wpeel',
#                         bcal_dir='/lustre/mmanders/exoplanet/BCAL/20200123_multiintStokesV_take2')
utc_times_txt_path='/lustre/mmanders/exoplanet/orca_test/LST_wpeel/utc_times.txt'
dadafile_dir='/lustre/data/exoplanet_20200117'
msfile_dir='/lustre/mmanders/exoplanet/orca_test/LST_wpeel'
bcal_dir='/lustre/mmanders/exoplanet/BCAL/20200123_multiintStokesV_take2'
flag_dir='/home/mmanders/imaging_scripts/flagfiles/20200122'
dadafilelist = np.genfromtxt(utc_times_txt_path, delimiter=' \t ', unpack=True, dtype=str)[1]

@app.task
def exoplanet_pipeline(dadafile = str, subband = int):
    spw    = '%02d' % subband
    # dada2ms
    msfile = run_dada2ms(dadafile_dir+'/'+spw+'/'+dadafile, \
                         msfile_dir+'/'+spw+'-'+path.splitext(dadafile)[0]+'.ms')
    # applycal
    bcal    = bcal_dir+'/'+spw+'-T1al.bcal'
    Xcal    = bcal_dir+'/'+spw+'-T1al.X'
    dcal    = bcal_dir+'/'+spw+'-T1al.dcal'
    #cygacal = bcal_dir+'/'+spw+'-T1al-spec.bcal'
    #applycal(msfile, gaintable=[bcal,Xcal,dcal,cygacal], flagbackup=False)
    applycal(msfile, gaintable=[bcal,Xcal,dcal], flagbackup=False)
    # generate and apply channel flags
    flag_chans(msfile, spw)
    # apply antenna flags
    ants = np.genfromtxt(flag_dir+'/all.antflags', dtype=int, delimiter=',')
    apply_ant_flag(msfile, ants)
    # apply baseline flags
    blfile = flag_dir+'/all.blflags'
    apply_bl_flag(msfile,blfile)
    # polarized peel
    zest(msfile)
    return msfile