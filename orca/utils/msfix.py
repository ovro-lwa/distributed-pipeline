#!/usr/bin/env python
"""Measurement set repair utilities.

Provides functions for fixing common issues with measurement sets
after concatenation, such as inconsistent FIELD_ID values.
"""
from __future__ import division
import numpy as np
import pyrap.tables as pt
import os,argparse


def concat_issue_fieldid(msfile: str, obsid: bool = False):
    """Fix FIELD_ID and optionally OBSERVATION_ID after CASA concat.

    Sets all FIELD_ID (and optionally OBSERVATION_ID) values to 0,
    which is required for WSClean to work with concatenated MSs.

    Args:
        msfile: Path to the measurement set.
        obsid: If True, also reset OBSERVATION_ID to 0.
    """
    t       = pt.table(msfile, readonly=False)
    fid     = t.getcol('FIELD_ID')
    fidnew  = np.zeros(fid.shape,dtype=int)
    t.putcol('FIELD_ID', fidnew)
    if obsid:
        oid    = t.getcol('OBSERVATION_ID')
        oidnew = np.zeros(oid.shape,dtype=int)
        t.putcol('OBSERVATION_ID', oidnew)
    t.close()

def main():
    parser = argparse.ArgumentParser(description="Sets Field ID for all integrations to 0. For using WSClean with MS concatenated using CASA concat.")
    parser.add_argument("msfile", help="Measurement set.")
    parser.add_argument("--obsid", action='store_true', default=False, help="If option is set, will also set Observation ID for all integrations to 0.")
    args = parser.parse_args()
    concat_issue_fieldid(args.msfile,args.obsid)

if __name__ == '__main__':
    main()
            
