import pytest
from mock import patch, Mock
import shlex
import subprocess

from orca.wrapper import wsclean


@patch('subprocess.Popen')
def test_wsclean(mocked_Popen):
    mocked_p = Mock()
    mocked_Popen.return_value = mocked_p
    mocked_p.communicate.return_value = (b'a', b'b')
    mocked_p.returncode = 0

    wsclean.wsclean(['test.ms'], '.', 'test', ['-size', '4096', '4096', '-weight', '0'])
    # wsclean now adds -j and -abs-mem arguments by default
    mocked_Popen.assert_called_once_with(
        shlex.split('/opt/bin/wsclean -j 1 -abs-mem 50 -size 4096 4096 -weight 0 -name ./test test.ms'),
        env=wsclean.NEW_ENV)


@patch('subprocess.Popen')
def test_wsclean_subprocess_error(mocked_Popen):
    mocked_p = Mock()
    mocked_Popen.return_value = mocked_p
    mocked_p.communicate.return_value = (b'a', b'b')
    mocked_p.returncode = 1
    with pytest.raises(Exception) as exinfo:
        wsclean.wsclean(['test.ms'], '.', 'test', ['-size', '4096', '4096'])

