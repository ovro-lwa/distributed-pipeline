import pytest
from mock import patch, Mock
from orca.wrapper import wsclean

@patch('subprocess.Popen')
def test_wsclean(mocked_Popen):
    mocked_p = Mock()
    mocked_Popen.return_value = mocked_p
    mocked_p.communicate.return_value = (b'a', b'b')
    mocked_p.returncode = 0

    wsclean.wsclean(['test.ms'], '.', 'test', ['-size', '4096', '4096'])
    mocked_Popen.assert_called_once_with([wsclean.WSCLEAN_1_11_EXEC, '-size',
                                          '4096', '4096', '-name', './test', 'test.ms'], env=wsclean.NEW_ENV)

@patch('subprocess.Popen')
def test_wsclean(mocked_Popen):
    mocked_p = Mock()
    mocked_Popen.return_value = mocked_p
    mocked_p.communicate.return_value = (b'a', b'b')
    mocked_p.returncode = 1
    with pytest.raises(Exception) as exinfo:
        wsclean.wsclean(['test.ms'], '.', 'test', ['-size', '4096', '4096'])

