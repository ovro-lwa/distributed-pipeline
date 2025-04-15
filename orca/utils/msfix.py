#!/usr/bin/env python

from __future__ import division
import numpy as np
import pyrap.tables as pt
import os,argparse

def concat_issue_fieldid(msfile,obsid=False):
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
            
