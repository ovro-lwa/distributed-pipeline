import logging

from orca.utils.sourcemodels import RFI_B, CYG_A_UNPOLARIZED_RESOLVED, CAS_A_UNPOLARIZED_RESOLVED
from orca.wrapper import ttcal
from typing import List, Optional
from datetime import datetime
import json

import casacore.tables as tables
from orca.utils import coordutils

log = logging.getLogger(__name__)
CORRECTED_DATA = 'CORRECTED_DATA'
DATA = 'DATA'


def ttcal_peel_from_data_to_corrected_data(ms: str, sources_json: str):
    with tables.table(ms, readonly=False) as t:
        # Copied from https://github.com/casacore/python-casacore/blob/master/casacore/tables/msutil.py#L48
        column_names = t.colnames()
        if CORRECTED_DATA not in column_names:
            dminfo = t.getdminfo(DATA)
            cdesc = t.getcoldesc(DATA)
            dminfo['NAME'] = 'correcteddata'
            cdesc['comment'] = 'The corrected data column'
            t.addcols(tables.maketabdesc(tables.makecoldesc(CORRECTED_DATA, cdesc)), dminfo)
            t.putcol(CORRECTED_DATA, t.getcol(DATA))
        else:
            log.info(f'{ms} already has {CORRECTED_DATA} column. Not peeling.')
            return ms
    # This reads from the just-created CORRECTED_DATA column and writes to CORRECTED_DATA column.
    ttcal.peel_with_ttcal(ms, sources_json)
    return ms


def write_peeling_sources_json(utc_timestamp: datetime, out_json: str, include_rfi_source: bool) -> Optional[str]:
    sources = _get_peeling_sources_dict(utc_timestamp, include_rfi_source)
    if sources:
        with open(out_json, 'w') as out_file:
            json.dump(sources, out_file)
        return out_json
    else:
        return None


def _get_peeling_sources_dict(utc_timestamp: datetime, include_rfi_source: bool) -> List[dict]:
    sources = []

    if coordutils.is_visible(coordutils.CYG_A, utc_timestamp):
        sources.append(
            CYG_A_UNPOLARIZED_RESOLVED)
    if coordutils.is_visible(coordutils.CAS_A, utc_timestamp):
        sources.append(
            CAS_A_UNPOLARIZED_RESOLVED
        )

    if include_rfi_source:
        sources.append(
            RFI_B
        )
    return sources
