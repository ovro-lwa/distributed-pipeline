import logging
from orca.wrapper import ttcal

log = logging.getLogger(__name__)


def peel_with_ttcal(ms: str, peel_with_rfi: bool):
    # Figure out the w
    ttcal.run_ttcal(ms, peel_with_rfi)
    return ms
