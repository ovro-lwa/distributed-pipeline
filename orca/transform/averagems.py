from casacore import tables
from typing import List
from os import path
import logging
log = logging.getLogger(__name__)

def average_ms(ms_list: List[str], ref_ms_index: int,  out_ms: str, column: str) -> str:
    """
    Average the list of measurement sets' select column.
    :param ms_list: the list of the measurement set.
    :param ref_ms_index: the index of the measurement set in ``ms_list`` to create the averaged measurement set from.
    :param out_ms: output measurement set path.
    :param column: Name of the column.
    :return:
    """
    count = float(len(ms_list))
    tables.tablecopy(ms_list[ref_ms_index], out_ms)
    logging.info('Reading reference measurement set.')
    with tables.table(out_ms, readonly=False) as out_table:
        averaged_data = out_table.getcol(column) / count
        for i, ms in enumerate(ms_list):
            if i != ref_ms_index:
                with tables.table(ms, readonly=True) as t:
                    averaged_data += t.getcol(column) / count
        out_table.putcol(column, averaged_data)
    return path.abspath(out_ms)
