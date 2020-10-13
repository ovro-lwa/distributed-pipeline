"""Wrapper on top of subprocess (under construction)
"""
from typing import List
import subprocess
import logging
log = logging.getLogger(__name__)


def check_output(cmd: List[str]) -> str:
    """Wrapper on top of subprocess.check_output.
    If success, returns the stdout.
    If failure, logs the stderr and throws CalledProcessError.
    shell=True is not allowed since it is not secure and does mess up how Popen handles errors.
    Args:
        cmd:

    Returns:

    """
    try:
        return subprocess.check_output(cmd, stderr=subprocess.PIPE).decode()
    except subprocess.CalledProcessError as e:
        log.error(f'Error while executing {str(cmd)}: stderr is {e.stderr}.')
        raise e
