import pytest
import numpy as np

from casacore.tables import table
import time

from orca.transform import calibration
from os import path

DATA_DIR = '/lustre/yuping/orca-test-resource'

@pytest.mark.skip(reason="numba dtype mismatch with complex64/float64 in current environment")
def test_applycal_in_mem():
    data = np.ones(shape=(3,1,4))
    bcal = np.array([ [ [1, 1/2] ],
                    [[1/3, 1/4]] ])
    print(bcal.shape)
    ans = calibration.applycal_in_mem(data, bcal)
    print(ans)
    assert (ans[0, 0] == np.array([1., 2., 2., 4.])).all()
    assert (ans[1, 0] == np.array([3., 4., 6., 8.])).all()
    assert (ans[2, 0] == np.array([9., 12., 12., 16.])).all()


@pytest.mark.skipif(not path.isdir(DATA_DIR), reason="need acual data.")
def test_on_ms():
    with table(f'{DATA_DIR}/20231120_104728_69MHz.ms', ack=False) as t:
        data = t.getcol('DATA')
        corrected = t.getcol('CORRECTED_DATA')
        d_flag = t.getcol('FLAG')

    with table(f'{DATA_DIR}/20231120_69MHz.bcal', ack=False) as t:
        bcal = t.getcol('CPARAM')
        flags = t.getcol('FLAG')
    
    start = time.time()
    out = calibration.applycal_in_mem(data, bcal)
    end = time.time()
    print("Elapsed (with compilation) = %s" % (end - start))

    diff = (np.abs(out - corrected))/np.abs(corrected)
    assert ((diff < 1e-6) | d_flag).all()

@pytest.mark.skipif(not path.isdir(DATA_DIR), reason="need acual data.")
def test_applycal_in_mem_cross_ms():
    with table(f'{DATA_DIR}/20231120_104728_69MHz.ms', ack=False) as t:
        tcross = t.query('ANTENNA1 != ANTENNA2')
        data = tcross.getcol('DATA')
        corrected = tcross.getcol('CORRECTED_DATA')
        d_flag = tcross.getcol('FLAG')

    with table(f'{DATA_DIR}/20231120_69MHz.bcal', ack=False) as t:
        bcal = t.getcol('CPARAM')
        flags = t.getcol('FLAG')
    
    out = calibration.applycal_in_mem_cross(data, bcal)
    diff = (np.abs(out - corrected))/np.abs(corrected)
    assert ((diff < 1e-6) | d_flag).all()