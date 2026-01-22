""" Wrappers for TTCal.
"""
import subprocess
import logging
import os

TTCAL_EXEC = '/opt/devel/pipeline/envs/julia060/bin/ttcal.jl'

def peel_with_ttcal(ms: str, sources: str):
    """Use TTCal to peel sources.
    
    Args:
        ms: Path to the measurement set.
        sources: Path to the sources.json file.
    
    Returns: The path to the measurement set because TTCal reads from and writes to it.
    """
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

def zest_with_ttcal(
    ms: str,
    sources: str = '/yourdirectory/sources.json',
    beam: str = 'constant',
    minuvw: int = 10,
    maxiter: int = 30,
    tolerance: str = '1e-4',
):
    """Use TTCal to run 'zest' with sensible defaults.

    Args:
        ms: Path to the measurement set.
        sources: Path to the sources.json file (default: /yourdirectory/sources.json).
        beam: TTCal beam model (default: 'constant').
        minuvw: Minimum uvw in wavelengths (default: 10).
        maxiter: Maximum iterations (default: 30).
        tolerance: Solver tolerance (default: '1e-4').

    Returns: The path to the measurement set (TTCal reads/writes in-place).
    """
    new_env = dict(os.environ, LD_LIBRARY_PATH='/opt/astro/mwe/usr/lib64:/opt/astro/lib/',
                   AIPSPATH='/opt/astro/casa-data dummy dummy')

    julia_path = '/opt/devel/pipeline/envs/julia060/bin/julia'

    proc = subprocess.Popen(
        [julia_path, TTCAL_EXEC, 'zest', ms, sources,
         '--beam', str(beam),
         '--minuvw', str(minuvw),
         '--maxiter', str(maxiter),
         '--tolerance', str(tolerance)],
        env=new_env,
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE
    )
    try:
        stdoutdata, stderrdata = proc.communicate()
        if proc.returncode != 0:
            logging.error(f'Error in TTCal zest: {stderrdata.decode()}')
            logging.info(f'stdout is {stdoutdata.decode()}')
            raise Exception('Error in TTCal zest.')
    finally:
        proc.terminate()
    return ms

