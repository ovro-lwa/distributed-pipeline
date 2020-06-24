import argparse

from orca.flagging import flagoperations


def main():
    parser = argparse.ArgumentParser(description="Baseline flagger.")
    parser.add_argument("msfile", help="Measurement set.")
    parser.add_argument("blfile", help="List of baseline flags. Expected format: one baseline per line, of the form 'ant1&ant2'. 0-indexed.")
    args = parser.parse_args()
    flagoperations.flag_bls(args.msfile, args.blfile)


if __name__ == '__main__':
    main()
