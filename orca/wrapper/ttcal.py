"""TTCal peeling and calibration wrapper.

Provides Python interface to TTCal, a Julia-based direction-dependent
calibration tool for peeling bright sources from visibility data.
Supports both standard peeling and polarized (ZEST) peeling modes.
"""
import subprocess
import logging
import os
import json
import shlex
import tempfile
from typing import Optional

import numpy as np

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
    solutions_path: Optional[str] = None,
    julia_env: str = 'julia060',
):
    """Use TTCal to run 'zest' with sensible defaults.

    Args:
        ms: Path to the measurement set.
        sources: Path to the sources.json file (default: /yourdirectory/sources.json).
        beam: TTCal beam model (default: 'constant').
        minuvw: Minimum uvw in wavelengths (default: 10).
        maxiter: Maximum iterations (default: 30).
        tolerance: Solver tolerance (default: '1e-4').
        solutions_path: Save compressed full-Jones solutions and provenance here.
            None preserves the legacy command. Phase 1 supplies this by default.
        julia_env: Existing conda environment name for solution-export runs.

    Returns: The path to the measurement set (TTCal reads/writes in-place).
    """
    if solutions_path is not None:
        return _zest_with_solutions(ms, sources, beam, minuvw, maxiter,
                                    tolerance, solutions_path, julia_env)
    if julia_env != 'julia060':
        raise ValueError('julia_env requires solutions_path')
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


def _zest_with_solutions(ms, sources, beam, minuvw, maxiter, tolerance,
                         solutions_path, julia_env):
    """Export using installed TTCal; package exchange files without Julia NPZ."""
    output = os.path.abspath(solutions_path)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(sources) as stream:
        source_model = json.load(stream)
    script = os.path.join(os.path.dirname(__file__), 'ttcal_solutions.jl')
    with tempfile.TemporaryDirectory(prefix='.peelsol-', dir=os.path.dirname(output)) as tmp:
        cmd = [f'/opt/devel/pipeline/envs/{julia_env}/bin/julia', script,
               os.path.abspath(ms), os.path.abspath(sources), str(beam),
               str(minuvw), str(maxiter), str(tolerance), tmp]
        env = os.environ.copy()
        if julia_env == 'julia060':
            env.update(LD_LIBRARY_PATH='/opt/astro/mwe/usr/lib64:/opt/astro/lib/',
                       AIPSPATH='/opt/astro/casa-data dummy dummy')
        else:
            # Preserve the existing Phase 1 RFI environment activation.
            env['OMP_NUM_THREADS'] = '8'
            cmd = ['/bin/bash', '-c',
                   'source ~/.bashrc && conda activate ' + shlex.quote(julia_env)
                   + ' && exec ' + ' '.join(shlex.quote(arg) for arg in cmd)]
        subprocess.run(cmd, env=env, check=True)
        with open(os.path.join(tmp, 'metadata.json')) as stream:
            metadata = json.load(stream)
        shape = tuple(metadata.pop('shape'))
        gains = np.fromfile(os.path.join(tmp, 'gains.bin'), dtype='<c16').reshape(shape, order='F')
        metadata.update(schema_version=1, ms=os.path.basename(os.path.normpath(ms)),
                        sources=source_model, beam=beam, minuvw=minuvw, maxiter=maxiter,
                        tolerance=float(tolerance), peeliter=3, routine='zest',
                        julia_env=julia_env,
                        gain_axes=['source', 'jones', 'antenna', 'frequency', 'time'],
                        convergence_flags_available=False)
        pending = os.path.join(tmp, 'solutions.npz')
        np.savez_compressed(
            pending, gains=gains, invalid_gains=~np.isfinite(gains).all(axis=1),
            source_indices=np.asarray(metadata['source_indices'], dtype=np.int64),
            source_names=np.asarray(metadata['source_names'], dtype=str),
            frequencies_hz=np.asarray(metadata['frequencies_hz'], dtype=np.float64),
            times_mjd_seconds=np.asarray(metadata['times_mjd_seconds'], dtype=np.float64),
            antenna_indices=np.arange(shape[2]), jones_order=np.array(['xx', 'xy', 'yx', 'yy']),
            metadata_json=np.array(json.dumps(metadata)),
        )
        os.replace(pending, output)
    logging.info('Saved peeling solutions %s (%d bytes)', output, os.path.getsize(output))
    return ms
