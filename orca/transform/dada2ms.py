from datetime import datetime
import logging
from typing import List, Dict
from os import path
import subprocess

# TODO turn this into a class and pull theses things into a config.
dada2ms_exec = '/opt/astro/dada2ms/bin/dada2ms-tst3'
dada2ms_config = '/home/yuping/dada2ms.cfg'


def get_ms_name(timestamp: datetime, spw: str, out_dir: str) -> str:
    """
    This should be managed by a different class (SubdirConvention or something).
    :param timestamp:
    :param spw:
    :param out_dir:
    :return:
    """
    return f'{out_dir}/{timestamp.isoformat()}/{spw}_{timestamp.isoformat()}.ms'


def run_dada2ms(dada_file: str, out_ms: str, gaintable: str = None) -> str:
    """
    Turns dada into ms.
    TODO generate python binding for dada2ms
    :param dada_file: Path to the dada file.
    :param out_ms: Path to the output measurement set.
    :param gaintable:
    :return: Path to the generated measurement set.
    TODO Generate a python binding for dada2ms and call it here,
    """
    if gaintable:
        subprocess.check_output(
            [dada2ms_exec, '-c', dada2ms_config, '--cal', gaintable, dada_file, out_ms], stderr=subprocess.STDOUT)
    else:
        subprocess.check_output(
            [dada2ms_exec, '-c', dada2ms_config, dada_file, out_ms], stderr=subprocess.STDOUT)
    return path.abspath(out_ms)


def generate_params(utc_mapping: Dict[datetime, str], begin_time: datetime, end_time: datetime,
                    spw: List[str], dada_prefix: str, out_dir: str) -> List[dict]:
    """

    :param utc_mapping:
    :param begin_time:
    :param end_time:
    :param spw: list of spectral windows
    :param dada_prefix prefix to paths of the dada files
    :param out_dir:
    :return:
    """
    return [{'dada_file': f'{dada_prefix}/{s}/{dada_file}',
             'out_ms': get_ms_name(time, s, out_dir),
             'gaintable': f'/lustre/yuping/0-100-hr-reduction/bcal/{s}_2018-03-22T17:34:35.bcal'} for s in spw
            for time, dada_file in utc_mapping.items() if begin_time <= time < end_time]

