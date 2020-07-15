"""Peeling related transforms
"""
import logging

from orca.utils.sourcemodels import RFI_B, CYG_A_UNPOLARIZED_RESOLVED, CAS_A_UNPOLARIZED_RESOLVED
from orca.wrapper import ttcal
from typing import List, Optional
from datetime import datetime
import tempfile
from os import path
import json

import casacore.tables as tables
from orca.utils import coordutils

log = logging.getLogger(__name__)
CORRECTED_DATA = 'CORRECTED_DATA'
DATA = 'DATA'


def ttcal_peel_from_data_to_corrected_data(ms: str, utc_time: datetime, include_rfi_source: bool = True) -> str:
    """ Use TTCal to peel. Read from DATA column and write to CORRECTED_DATA
    If the CORRECTED_DATA column exists, it does not do anything.

    Args:
        ms: Path to measurement set.
        utc_time: datetime object to figure out what sources are up.
        include_rfi_source: Include near-field generic RFI sources in peel.

    Returns: The output measurement set (which is the same thing as the input measurement set).

    """
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
    log.info(f'Generating sources.json for {ms}')
    with tempfile.TemporaryDirectory() as tmpdir:
        sources_json = path.join(tmpdir, 'sources.json')
        if _write_peeling_sources_json(utc_time, sources_json, include_rfi_source=include_rfi_source):
            # This reads from the just-created CORRECTED_DATA column and writes to CORRECTED_DATA column.
            ttcal.peel_with_ttcal(ms, sources_json)
    return ms


def _write_peeling_sources_json(utc_timestamp: datetime, out_json: str, include_rfi_source: bool) -> Optional[str]:
    sources = _get_peeling_sources_list(utc_timestamp, include_rfi_source)
    if sources:
        log.info(f'{len(sources)} sources to peel for {utc_timestamp.isoformat()}.')
        with open(out_json, 'w') as out_file:
            json.dump(sources, out_file)
        return out_json
    else:
        log.info(f'No sources to peel for {utc_timestamp.isoformat()}')
        return None


def _get_peeling_sources_list(utc_timestamp: datetime, include_rfi_source: bool) -> List[dict]:
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
