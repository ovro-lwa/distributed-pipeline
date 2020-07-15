"""Visibility subtraction (not tested)
"""
import argparse
import logging
from casacore.tables import table, tablecopy
import numpy as np


def diff(early_vis, late_vis, out_vis):
    late_vis_table = table(late_vis)
    diff_table = tablecopy(early_vis, out_vis)
    # Validating that the two vis sets have the same antenna pairs for each row
    assert np.equal(late_vis_table.getcol('ANTENNA1'), diff_table.getcol('ANTENNA1')).all(), \
        'The visibility files are ordered differently'
    assert np.equal(late_vis_table.getcol('ANTENNA2'), diff_table.getcol('ANTENNA2')).all(), \
        'The visibility files are ordered differently'
    diff_vis = late_vis_table.getcol('CORRECTED_DATA') - diff_table.getcol('CORRECTED_DATA')
    diff_table.close()
    diff_table = table(out_vis, readonly=False)
    diff_table.putcol('DATA', diff_vis)
    diff_table.putcol('CORRECTED_DATA', diff_vis)
    late_vis_table.close()
    diff_table.close()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.description = """
    Subtract two sets of visibilities and write the output to a output visibility file.
    The metadata (and uvw) of the output comes from the early visibility file.
    """
    parser.add_argument('early_vis_prefix', help='The early visibility prefix.')
    parser.add_argument('late_vis_prefix', help='The late visibility prefix.')
    parser.add_argument('output_vis_dir', help='The directory of the output visibility.')
    args = parser.parse_args()
    for i in range(22):
        spw = f'{i:02d}'
        logging.info(f"diff'ing spw {spw}")
        diff(f'{args.early_vis_prefix}/{spw}/{spw}_{args.early_vis_prefix}.ms',
             f'{args.late_vis_prefix}/{spw}/{spw}_{args.late_vis_prefix}.ms',
             f'{args.output_vis_dir}/{spw}.ms')
