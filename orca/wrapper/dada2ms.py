"""dada2ms wrapper
"""
import logging
from os import path
import os
import subprocess

# TODO turn this into a class and pull theses things into a config.
dada2ms_exec = '/opt/astro/dada2ms/bin/dada2ms-tst3'
dada2ms_config = '/home/yuping/dada2ms.cfg'

NEW_ENV = new_env = dict(os.environ, LD_LIBRARY_PATH='/opt/astro/mwe/usr/lib64:/opt/astro/lib/:/opt/astro/casacore-1.7.0/lib',
                         AIPSPATH='/opt/astro/casa-data dummy dummy')

def dada2ms(dada_file: str, out_ms: str, gaintable: str = None) -> str:
    """Wrapper around Stephen Bourke's dada2ms.
    Optionally apply a gaintable (only gaintable of type bandpass has been tested). It will write a ms with the data in
    the DATA column. If the directory of the out_ms does not exist, it will create the directory.

    Args:
        dada_file: Path to the dada file.
        out_ms: Path to the output measurement set.
        gaintable: Path to the gaintable. Default is None which means don't apply the calibration.

    Returns: Path to the output ms. The same as out_ms

    """
    os.makedirs(out_ms, exist_ok=True)

    if gaintable:
        proc = subprocess.Popen(
            [dada2ms_exec, '-c', dada2ms_config, '--cal', gaintable, dada_file, out_ms], env=NEW_ENV,
            stderr=subprocess.PIPE, stdout=subprocess.PIPE
        )
    else:
        proc = subprocess.Popen(
            [dada2ms_exec, '-c', dada2ms_config, dada_file, out_ms], env=NEW_ENV, stderr=subprocess.PIPE,
            stdout=subprocess.PIPE)
    stdoutdata, stderrdata = proc.communicate()
    if proc.returncode is not 0:
        logging.error(f'Error in data2ms: {stderrdata.decode()}')
        raise Exception('Error in dada2ms.')
    return path.abspath(out_ms)

