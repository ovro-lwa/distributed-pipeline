"""Command-line entry point for planning, running, and inspecting jobs."""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import List, Optional

from .config import Config
from .ledger import Ledger
from .orchestrator import Orchestrator

def _worker_sh(config_dir: str) -> str:
    """Find the worker launcher beside the config or from the environment."""
    env = os.environ.get("GSI_WORKER_SH")
    if env:
        return env
    return str(Path(config_dir).resolve().parent / "workers" / "run_worker.sh")


log = logging.getLogger("gsi.cli")


def _freeze_config_dir(config_dir: str, cfg: Config) -> str:
    """Copy the YAML files so every worker sees the same run configuration."""
    src = Path(config_dir).resolve()
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    dst = (Path(cfg.cluster.state_dir) / "run_configs" /
           f"{cfg.pipeline.date}-{stamp}-{os.getpid()}")
    dst.mkdir(parents=True, exist_ok=False)
    for name in ("cluster.yaml", "subbands.yaml", "pipeline.yaml"):
        shutil.copy2(src / name, dst / name)
    log.info("using frozen config snapshot %s", dst)
    return str(dst)


def _orch(config_dir: str, list_node: Optional[str],
          freeze_config: bool = False) -> Orchestrator:
    cfg = Config.load(config_dir)
    worker_sh = _worker_sh(config_dir)
    worker_config_dir = (_freeze_config_dir(config_dir, cfg)
                         if freeze_config else config_dir)
    return Orchestrator(cfg, worker_config_dir, worker_sh, list_node)


def _config_dir(arg: str) -> str:
    p = Path(arg).resolve()  # Workers need an absolute shared path.
    return str(p if p.is_dir() else p.parent)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s",
                        force=True)
    ap = argparse.ArgumentParser(prog="gsi")
    ap.add_argument("cmd", choices=["run", "resume", "status", "dry-run", "stitch"])
    ap.add_argument("--config", required=True, help="config dir or a yaml in it")
    ap.add_argument("--list-node", default=None,
                    help="node used to list lustre (default: first slot's node)")
    ap.add_argument("--band", type=int, help="band for `stitch`")
    a = ap.parse_args(argv)

    cdir = _config_dir(a.config)
    logging.getLogger("gsi.cli").info("command=%s config=%s", a.cmd, cdir)

    if a.cmd == "status":
        cfg = Config.load(cdir)
        led = Ledger(Path(cfg.cluster.state_dir) / f"ledger_{cfg.pipeline.date}.sqlite")
        print(cfg.pipeline.date, led.summary())
        led.close()
        return 0

    if a.cmd == "dry-run":
        orch = _orch(cdir, a.list_node)
        orch.dry_run()
    elif a.cmd in ("run", "resume"):
        orch = _orch(cdir, a.list_node, freeze_config=True)
        orch.plan()  # Existing ledger state makes run and resume equivalent.
        orch.run()
    elif a.cmd == "stitch":
        if a.band is None:
            ap.error("--band required for stitch")
        orch = _orch(cdir, a.list_node, freeze_config=True)
        slot = orch.slots[0]
        from . import dispatch
        dispatch.run_worker(slot.node, slot.gpu, orch.worker_sh, orch.config_dir,
                           a.band, -1, "STITCH", env=orch.worker_env)
    return 0


if __name__ == "__main__":
    sys.exit(main())
