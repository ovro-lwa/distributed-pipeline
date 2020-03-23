import logging
from orca.wrapper import ttcal

import casacore.tables as pt

log = logging.getLogger(__name__)
CORRECTED_DATA = 'CORRECTED_DATA'
DATA = 'DATA'


def peel_with_ttcal(ms: str, sources_json: str):
    with pt.table(ms, readonly=False) as t:
        # Copied from https://github.com/casacore/python-casacore/blob/master/casacore/tables/msutil.py#L48
        column_names = t.colnames()
        if CORRECTED_DATA not in column_names:
            dminfo = t.getdminfo(DATA)
            cdesc = t.getcoldesc(DATA)
            dminfo['NAME'] = 'correcteddata'
            cdesc['comment'] = 'The corrected data column'
            t.addcols(pt.maketabdesc(pt.makecoldesc(CORRECTED_DATA, cdesc)), dminfo)
            t.putcol(CORRECTED_DATA, t.getcol(DATA))
        else:
            log.info(f'{ms} already has {CORRECTED_DATA} column. Not peeling.')
            return ms
    # corrected data column stuff
    ttcal.peel_with_ttcal(ms, sources_json)
    return ms
