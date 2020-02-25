import pytest
import subprocess
from orca.utils import commandline
import sys

# Tests to figure out how subprocess behave
BAD_COMMAND = ['ls', '/meh/']


def test_subprocess_check_output_return_stderr():
    with pytest.raises(subprocess.CalledProcessError) as exinfo:
        subprocess.check_output(BAD_COMMAND, stderr=subprocess.PIPE)
    assert exinfo.value.stderr == b"ls: cannot access '/meh/': No such file or directory\n"


def test_subprocess_check_call_no_stderr():
    with pytest.raises(subprocess.CalledProcessError) as exinfo:
        subprocess.check_call(BAD_COMMAND, stderr=subprocess.PIPE)
    assert exinfo.value.stderr is None


def test_subprocess_check_output():
    assert subprocess.check_output(['echo', '1'], stderr=subprocess.PIPE) == b"1\n"


def test_subprocess_check_output_error_with_stdout_and_stderr():
    # TODO I can't think of a good command here. This command should write something to stdout and then errors.
    # Then I should check for both stdout and stderr in the resultant CalledProcessError
    pytest.skip("Can't think of a test case yet.")


def test_shell_equal_true_messes_up_error_catching():
    subprocess.check_call(BAD_COMMAND, shell=True)


def test_commandline_check_output_gives_output():
    assert commandline.check_output(['echo', '1']) == "1\n"


def test_commandline_check_output_return_stderr():
    with pytest.raises(subprocess.CalledProcessError) as exinfo:
        commandline.check_output(BAD_COMMAND)
    assert exinfo.value.stderr == b"ls: cannot access '/meh/': No such file or directory\n"
