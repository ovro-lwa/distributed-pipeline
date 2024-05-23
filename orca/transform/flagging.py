from typing import List

import subprocess
import logging
from casacore import tables

from orca.configmanager import execs

log = logging.getLogger(__name__)

def flag_with_aoflagger(ms: str, strategy: str='/opt/share/aoflagger/strategies/nenufar-lite.lua') -> str:
    # TODO use the API
    arg_list = [execs.aoflagger, '-strategy', strategy, ms]
    proc = subprocess.Popen(arg_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        stdoutdata, stderrdata = proc.communicate()
        if proc.returncode != 0:
            if stderrdata:
                log.error(f'Error in aoflagger: {stderrdata.decode()}')
            raise RuntimeError(f'Error in aoflagger for {ms}.')
    finally:
        proc.terminate()
    return ms


def flag_ants(ms: str, ants: List[int]) -> str:
    """
    Input: msfile, list of antennas to flag
    Flags the antennas in the list.
    """
    if len(ants) > 0 :
        tables.taql('UPDATE %s SET FLAG = True WHERE ANTENNA1 IN %s OR ANTENNA2 IN %s' % (ms, tuple(ants), tuple(ants)))
    return ms