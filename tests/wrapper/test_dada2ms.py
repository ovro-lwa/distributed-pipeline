import pytest
from mock import patch, Mock
import shlex
import subprocess

from orca.wrapper import dada2ms


@patch('os.makedirs')
@patch('subprocess.Popen')
def test_dada2ms_no_gaintable(mocked_Popen, mocked_makedirs):
    mocked_p = Mock()
    mocked_Popen.return_value = mocked_p
    mocked_p.communicate.return_value = (b'a', b'b')
    mocked_p.returncode = 0

    dada2ms.dada2ms('/test/data/1234567.dada', '/out/ms/out.ms')
    mocked_Popen.assert_called_once_with(
        shlex.split(f'{dada2ms.dada2ms_exec} -c {dada2ms.dada2ms_config} /test/data/1234567.dada /out/ms/out.ms'),
        env=dada2ms.NEW_ENV, stderr=subprocess.PIPE, stdout=subprocess.PIPE)


@patch('os.makedirs')
@patch('subprocess.Popen')
def test_dada2ms_with_gaintable(mocked_Popen, mocked_makedirs):
    mocked_p = Mock()
    mocked_Popen.return_value = mocked_p
    mocked_p.communicate.return_value = (b'a', b'b')
    mocked_p.returncode = 0

    dada2ms.dada2ms('/test/data/1234567.dada', '/out/ms/out.ms', gaintable='/path/to/a1.bcal')
    mocked_Popen.assert_called_once_with(
        shlex.split(f'{dada2ms.dada2ms_exec} -c {dada2ms.dada2ms_config} --cal /path/to/a1.bcal '
                    f'/test/data/1234567.dada /out/ms/out.ms'),
        env=dada2ms.NEW_ENV, stderr=subprocess.PIPE, stdout=subprocess.PIPE)


@patch('os.makedirs')
@patch('subprocess.Popen')
def test_make_dirs_called(mocked_Popen, mocked_makedirs):
    mocked_p = Mock()
    mocked_Popen.return_value = mocked_p
    mocked_p.communicate.return_value = (b'a', b'b')
    mocked_p.returncode = 0

    dada2ms.dada2ms('/test/data/1234567.dada', '/out/ms/out.ms', gaintable='/path/to/a1.bcal')
    mocked_makedirs.assert_called_once_with('/out/ms/out.ms', exist_ok=False)


@patch('logging.error')
@patch('os.makedirs')
@patch('subprocess.Popen')
def test_dada2ms_with_error(mocked_Popen, mocked_makedirs, mocked_logging):
    mocked_p = Mock()
    mocked_Popen.return_value = mocked_p
    mocked_p.communicate.return_value = (b'a', b'b')
    mocked_p.returncode = 1
    with pytest.raises(Exception) as exinfo:
        dada2ms.dada2ms('/test/data/1234567.dada', '/out/ms/out.ms')
        mocked_logging.assert_called_once()
