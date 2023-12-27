import pytest
from mock import patch, Mock
import subprocess

from orca.wrapper import change_phase_centre


@patch('subprocess.Popen')
def test_change_phase_center(mocked_Popen):
    mocked_p = Mock()
    mocked_Popen.return_value = mocked_p
    mocked_p.communicate.return_value = (b'a', b'b')
    mocked_p.returncode = 0

    change_phase_centre.change_phase_center('/test/data/1234567.ms', '09h18m05.8s -12d05m44s')
    mocked_Popen.assert_called_once_with(
        [change_phase_centre.CHGCENTRE_PATH, '/test/data/1234567.ms', '09h18m05.8s', '-12d05m44s'])
