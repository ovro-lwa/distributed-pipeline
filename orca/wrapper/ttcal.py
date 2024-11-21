""" Wrappers for TTCal.
"""
import subprocess
import logging
import os

TTCAL_EXEC = '/opt/devel/pipeline/envs/julia060/bin/ttcal.jl'

def peel_with_ttcal(ms: str, sources: str):
    """Use TTCal to peel sources."""
    new_env = dict(os.environ, LD_LIBRARY_PATH='/opt/astro/mwe/usr/lib64:/opt/astro/lib/',
                   AIPSPATH='/opt/astro/casa-data dummy dummy')

    julia_path = '/opt/devel/pipeline/envs/julia060/bin/julia'

    proc = subprocess.Popen(
        [julia_path, TTCAL_EXEC, 'peel', ms, sources, '--beam', 'sine', '--maxiter', '50',
         '--tolerance', '1e-4', '--minuvw', '10'],
        env=new_env,
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE
    )
    try:
        stdoutdata, stderrdata = proc.communicate()
        if proc.returncode != 0:
            logging.error(f'Error in TTCal: {stderrdata.decode()}')
            logging.info(f'stdout is {stdoutdata.decode()}')
            raise Exception('Error in TTCal.')
    finally:
        proc.terminate()
    return ms

