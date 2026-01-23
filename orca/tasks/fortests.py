"""Test and debugging tasks for Celery verification.

Contains simple Celery tasks for testing worker functionality,
task routing, and basic operations. These are not used in production
pipelines.
"""
from orca.celery import app

import logging
import os, shutil

import billiard

logging.getLogger(__name__)


import shutil
import os
import numpy as np
from casacore.tables import table


@app.task
def add(x: int, y: int) -> int:
    """Simple addition task for testing Celery connectivity.

    Args:
        x: First integer.
        y: Second integer.

    Returns:
        Sum of x and y.
    """
    return x+y


@app.task
def str_concat(first: str, second: str, third: str = '') -> str:
    """String concatenation task for testing.

    Args:
        first: First string.
        second: Second string.
        third: Optional third string.

    Returns:
        Concatenated string.
    """
    return f'{first}{second}{third}'


@app.task
def pcp(source: str, target_dir: str):
    """Copy a file to a target directory.

    Args:
        source: Source file path.
        target_dir: Target directory path.
    """
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(source, target_dir)


@app.task
def pcp_tree(source: str, target_dir: str):
    """Copy a directory tree to a target location.

    Args:
        source: Source directory path.
        target_dir: Target parent directory.
    """
    os.makedirs(target_dir, exist_ok=True)
    shutil.copytree(source, f'{target_dir}/{os.path.basename(source)}')


@app.task
def read_ms(ms: str, lock: bool):
    """Test reading a measurement set.

    Args:
        ms: Path to measurement set.
        lock: If True, open read-only.
    """
    with table(ms, ack=False, readonly=lock) as t:
        arr = t.getcol('DATA')


@app.task
def test_multiprocessing():
    """Test billiard multiprocessing in Celery workers."""
    def worker(num):
        return num

    for i in range(5):
        p = billiard.Process(target=worker, args=(i,))
        p.start()


@app.task
def do_sum(args):
    """Sum a sequence of numbers.

    Args:
        args: Iterable of numbers.

    Returns:
        Sum of the numbers.
    """
    s = 0
    for i in args:
        s += i
    return s