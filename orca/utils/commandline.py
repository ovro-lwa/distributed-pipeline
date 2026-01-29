"""Command-line execution utilities.

Provides wrappers around subprocess for executing external commands
with improved error handling and logging.
"""
from typing import List
import subprocess
import logging
log = logging.getLogger(__name__)


def check_output(cmd: List[str]) -> str:
    """Execute a command and return its stdout.

    Wrapper around subprocess.check_output with logging on failure.
    Does not use shell=True for security.

    Args:
        cmd: Command and arguments as a list of strings.

    Returns:
        Decoded stdout from the command.

    Raises:
        subprocess.CalledProcessError: If the command returns non-zero.
    """
    try:
        return subprocess.check_output(cmd, stderr=subprocess.PIPE).decode()
    except subprocess.CalledProcessError as e:
        log.error(f'Error while executing {str(cmd)}: stderr is {e.stderr}.')
        raise e
