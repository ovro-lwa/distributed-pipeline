import pytest
from orca.transform import dada2ms
from datetime import datetime


def test_generate_params():
    utc_mapping = {datetime(2008, 1, 1, 12, 00, 00): '2008_0001.dada',
                   datetime(2008, 1, 1, 13, 00, 00): '2008_0002.dada',
                   datetime(2008, 1, 1, 13, 00, 13): '2008_0003.dada',
                   datetime(2008, 1, 1, 13, 00, 26): '2008_0004.dada',
                   datetime(2008, 1, 1, 13, 10, 59): '2008_0005.dada'}
    n_spw = 22
    spw = [f'{i:02d}' for i in (range(n_spw))]
    output = dada2ms.generate_params(utc_mapping, datetime(2008, 1, 1, 13, 00, 00),
                                     datetime(2008, 1, 1, 13, 1, 0), spw, 'dada_prefix', 'outdir')
    dadas = [kwarg['dada_file'] for kwarg in output]
    assert 'dada_prefix/00/2008_0001.dada' not in dadas
    assert 'dada_prefix/00/2008_0005.dada' not in dadas
    assert 'dada_prefix/00/2008_0002.dada' in dadas
    assert len(dadas) == 3 * n_spw
