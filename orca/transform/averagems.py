"""Average measurement sets.
"""
from casacore import tables
from typing import List
from os import path
import logging
log = logging.getLogger(__name__)


def average_ms(ms_list: List[str], ref_ms_index: int, out_ms: str, column: str, tolerate_ms_io_error: bool = False) \
        -> str:
    """ Average the list of measurement sets' select column.
    Args:
        ms_list: the list of the measurement set.
        ref_ms_index: the index of the measurement set in ``ms_list`` to create the averaged measurement set from.
        out_ms: output measurement set path.
        column: Name of the column.
        tolerate_ms_io_error: Skip a measurement set if an exception was raised while trying to access it.

    Returns: Path to the averaged measurement set.

    """
    count = float(len(ms_list))
    tables.tablecopy(ms_list[ref_ms_index], out_ms)
    log.info('Reading reference measurement set.')
    with tables.table(out_ms, readonly=False) as out_table:
        averaged_data = out_table.getcol(column) / count
        for i, ms in enumerate(ms_list):
            if i != ref_ms_index:
                try:
                    with tables.table(ms, readonly=True) as t:
                        averaged_data += t.getcol(column) / count
                except RuntimeError as e:
                    if tolerate_ms_io_error:
                        log.exception('Skipping IO Error from measurement set:')
                    else:
                        raise e
        out_table.putcol(column, averaged_data)
    return path.abspath(out_ms)
