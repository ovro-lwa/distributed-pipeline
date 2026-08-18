"""Run remote workers and report node load over SSH."""
from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from typing import Mapping, Optional

_SSH = ["ssh", "-o", "ConnectTimeout=20", "-o", "BatchMode=yes"]


@dataclass
class RunResult:
    ok: bool
    rc: int
    tail: str  # Tail of combined output for error reports.


def _ssh(node: str, remote_cmd: str, timeout: Optional[int]) -> RunResult:
    try:
        p = subprocess.run(_SSH + [node, remote_cmd], capture_output=True,
                           text=True, timeout=timeout)
        out = (p.stdout + p.stderr)
        return RunResult(p.returncode == 0, p.returncode, out[-1500:])
    except subprocess.TimeoutExpired:
        return RunResult(False, -1, "ssh timeout")


def node_free_cores(node: str) -> Optional[float]:
    """Return `nproc - load1`, or `None` when the node is unreachable."""
    r = _ssh(node, "echo $(nproc) $(cut -d' ' -f1 /proc/loadavg)", timeout=25)
    if not r.ok:
        return None
    try:
        nproc, load1 = r.tail.split()
        return float(nproc) - float(load1)
    except ValueError:
        return None


def run_worker(node: str, gpu: int, worker_sh: str, config_dir: str,
               band: int, batch: int, manifest: str,
               timeout: Optional[int] = None,
               env: Optional[Mapping[str, str]] = None) -> RunResult:
    """Run one batch using a shared newline-delimited manifest."""
    args = [worker_sh, str(gpu), config_dir, str(band), str(batch), manifest]
    if env:
        args = ["env", *(f"{key}={value}" for key, value in env.items()), *args]
    cmd = " ".join(shlex.quote(a) for a in args)
    return _ssh(node, cmd, timeout=timeout)
