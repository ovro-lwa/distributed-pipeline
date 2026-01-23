"""Visibility subtraction utilities.

Provides functions for computing difference between visibility datasets,
useful for transient detection and background subtraction.

Note:
    This module is experimental and not fully tested.
"""
import argparse
import logging
from casacore.tables import table, tablecopy
import numpy as np


def diff(early_vis: str, late_vis: str, out_vis: str):
    """Compute difference between two visibility files.

    Subtracts early_vis from late_vis and writes to out_vis.
    Output metadata (including uvw) comes from early_vis.

    Args:
        early_vis: Path to the earlier visibility file.
        late_vis: Path to the later visibility file.
        out_vis: Path for the output difference visibility file.

    Raises:
        AssertionError: If the visibility files have different antenna ordering.
    """
    late_vis_table = table(late_vis, ack=False)
    diff_table = tablecopy(early_vis, out_vis)
    # Validating that the two vis sets have the same antenna pairs for each row
    assert np.equal(late_vis_table.getcol('ANTENNA1'), diff_table.getcol('ANTENNA1')).all(), \
        'The visibility files are ordered differently'
    assert np.equal(late_vis_table.getcol('ANTENNA2'), diff_table.getcol('ANTENNA2')).all(), \
        'The visibility files are ordered differently'
    diff_vis = late_vis_table.getcol('CORRECTED_DATA') - diff_table.getcol('CORRECTED_DATA')
    diff_table.close()
    diff_table = table(out_vis, readonly=False, ack=False)
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
