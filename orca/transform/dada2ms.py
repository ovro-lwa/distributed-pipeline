from astropy.time import Time
from typing import List
from os import path
import subprocess

# TODO turn this into a class and pull theses things into a config.
dada2ms_exec = '/opt/astro/dada2ms/bin/dada2ms-tst3'
dada2ms_config = '/lustre/yuping/0-common/dada2ms.cfg'


def get_ms_name(timestamp: Time, spw: int, out_dir: str) -> str:
    return f'{out_dir}/{spw}/{timestamp.value}'


def dada2ms(dada_file: str, out_ms: str) -> str:
    """
    Turns dada into ms.
    TODO generate python binding for dada2ms
    :param dada_file: Path to the dada file.
    :param out_ms: Path to the output measurement set.
    :return: Path to the generated measurement set.
    TODO Generate a python binding for dada2ms and call it here,
    """
    subprocess.check_output([dada2ms_exec, '-c', dada2ms_config, dada_file, out_ms])
    return path.abspath(out_ms)


def batch_dada2ms(utc_mapping: dict, utc_timestamps: Time, out_dir: str) -> List[str]:
    """
    dada2ms all dada files within time range and spws given. This assumes a certain directory structure
    for the data dump.
    :param utc_mapping: Mapping from utc timestamp to dada files
    :param utc_timestamps: Time range of desired observation
    :param out_dir: Output directory
    :return: List of paths to generated measurement sets.
    """
    pass
