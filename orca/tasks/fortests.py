from orca.celery import app

import logging
import os, shutil

import billiard

logging.getLogger(__name__)


"""
For debugging
"""

import shutil
import os
import numpy as np
from casacore.tables import table

@app.task
def add(x: int, y: int) -> int:
    return x+y

@app.task
def str_concat(first, second, third=''):
    return f'{first}{second}{third}'

@app.task
def pcp(source, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(source, target_dir)


@app.task
def pcp_tree(source, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    shutil.copytree(source, f'{target_dir}/{os.path.basename(source)}')


@app.task
def read_ms(ms, lock):
    with table(ms, ack=False, readonly=lock) as t:
        arr = t.getcol('DATA')


@app.task
def test_multiprocessing():
    def worker(num):
        return num

    for i in range(5):
        p = billiard.Process(target=worker, args=(i,))
        p.start()