"""Command-line interface for applying baseline flags.

Provides a CLI wrapper around flagoperations.flag_bls for applying
baseline flags from a text file to a measurement set.

Usage
-----
python flag_bls.py <msfile> <blfile>

See Also
--------
flagoperations.flag_bls : The underlying function that applies flags.
"""
import argparse

from orca.flagging import flagoperations


def main():
    """Parse arguments and apply baseline flags from file to MS."""
    parser = argparse.ArgumentParser(description="Baseline flagger.")
    parser.add_argument("msfile", help="Measurement set.")
    parser.add_argument("blfile", help="List of baseline flags. Expected format: one baseline per line, of the form 'ant1&ant2'. 0-indexed.")
    args = parser.parse_args()
    flagoperations.flag_bls(args.msfile, args.blfile)


if __name__ == '__main__':
    main()
