from tests import common
from tests.common import marin_gen_chanflags
import pytest
import numpy as np


ms_name = common.config['test_ms']


def test_get_marin_baseline():
    """
    Consistency check with Marin's code while I refactor gen_chanflags.py
    :return:
    """
    antchanflags, ant_flagged = marin_gen_chanflags.generate_chan_flags(ms_name)
    np.savetxt('../resources/marin_ant_chan_flags.txt', antchanflags, fmt='%i')
    np.savetxt('../resources/marin_ant_flagged.txt', ant_flagged, fmt='%i')


def test_the_whole_thing():
    """
    Consistency final check my code against Marin's
    :return:
    """
