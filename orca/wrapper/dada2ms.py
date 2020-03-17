from datetime import datetime
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
    """
    Turns dada into ms.
    TODO generate python binding for dada2ms
    :param dada_file: Path to the dada file.
    :param out_ms: Path to the output measurement set.
    :param gaintable:
    :return: Path to the generated measurement set.
    TODO Generate a python binding for dada2ms and call it here,
    """

    """
    TODO need to rebuild dada2ms and this won't be needed. Currently dada2ms depends on casacore libs ver 1.x and so
    I need to set the LD_LIBRARY_PATH to something else. Alternatively, we could add a symlink in the casacore 2.0 dir.
    I don't understand why I need this AIPSPATH stuff though.
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

