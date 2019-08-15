import pytest
from orca.transform import dada2ms
from datetime import datetime


def test_get_ms_name():
    t = datetime(2008, 1, 1, 23, 4, 3)
    assert dada2ms.get_ms_name(t, 0, '/lustre/yuping/2019-02-02-test-stuff') \
           == '/lustre/yuping/2019-02-02-test-stuff/2008-01-01T23:04:03/00_2008-01-01T23:04:03.ms'


def test_generate_params():
    utc_mapping = {datetime(2008, 1, 1, 12, 00, 00): '2008_0001.dada',
                   datetime(2008, 1, 1, 13, 00, 00): '2008_0002.dada',
                   datetime(2008, 1, 1, 13, 00, 13): '2008_0003.dada',
                   datetime(2008, 1, 1, 13, 00, 26): '2008_0004.dada',
                   datetime(2008, 1, 1, 13, 10, 59): '2008_0005.dada'}
    n_spw = 22
    spw = list(range(n_spw))
    output = dada2ms.generate_params(utc_mapping, datetime(2008, 1, 1, 13, 00, 00),
                                     datetime(2008, 1, 1, 13, 1, 0), spw, 'outdir')
    dadas = [kwarg['dada_file'] for kwarg in output]
    assert '2008_0001.dada' not in dadas
    assert '2008_0005.dada' not in dadas
    assert '2008_0002.dada' in dadas
    assert len(dadas) == 3 * n_spw
