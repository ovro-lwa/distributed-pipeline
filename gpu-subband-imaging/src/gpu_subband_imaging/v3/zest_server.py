"""v3.zest_persistent: one zest daemon per GPU that outlives the batch (no Julia JIT per batch).

Each band-hour batch is its own worker process, and its first peel compiles TTCalX (~40 s on the
critical path).  With the option, the daemon runs under a small server process per GPU and node
(``python -m gpu_subband_imaging.v3.zest_server <spec.json>``, its own session) that a batch attaches
to over a UNIX socket and detaches from at the end of its snapshot loop:

* attach (``PersistentZest.attach``): per-GPU flock; a live server with the same key (the daemon's
  command, its ZEST_* / JULIA_* / CUDA_* environment and the files it loads) is reused, else it is
  told to quit and a new one is started.  The flock is held while attached; a second batch on the
  same GPU (lanes, prefetch) does not wait for it but gets a private daemon (``None``).
* detach (``close``): RELEASE -- the daemon frees its workspace, device slot buffers and warm-start
  gains, collects and gives the pool back to the driver (``release_hard``: also destroys the CUDA
  context, so the idle daemon holds no GPU memory at all during the hour stage), then the
  connection is closed.  A client that disconnects without it (a killed worker) is released by
  the server.
* the server quits its daemon and exits after ``idle_s`` without a client, and on ``quit``.

The daemon is the ordinary ZestDaemon (timeouts, restarts, stale replies as before); its PEEL_* /
ZEST_TIMING log lines are returned with each reply and logged by the worker under gsi.peel.

Protocol: one JSON object per line each way.  Requests {"op": hello | zest | reset_warm | release |
quit, ...}; replies {"ok": true, ...} or {"ok": false, "error": str, "timeout": bool}.
"""
from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger("gsi.peel")

PROTOCOL = 1
# environment variables that change the daemon's behaviour: part of the server key
_KEY_ENV_PREFIXES = ("ZEST_", "JULIA_", "CUDA_", "TTCALX", "OPENBLAS_", "LD_LIBRARY_PATH", "PATH", "CASARC")


def base_dir() -> Path:
    return Path(os.environ.get("GSI_ZEST_SERVER_DIR") or f"/tmp/gsi_zest_{os.getuid()}")


def _file_sig(p: str) -> list:
    try:
        st = os.stat(p)
        return [p, st.st_size, int(st.st_mtime)]
    except OSError:
        return [p, None, None]


def server_key(cmd: List[str], run_env: Dict[str, str], files: List[str]) -> str:
    """Two batches may share a daemon iff this is equal: the daemon command, the environment that
    reaches its behaviour, and the size / mtime of the files it loads (daemon script, sky model,
    TTCalX sources, this module)."""
    env = {k: v for k, v in run_env.items() if k.startswith(_KEY_ENV_PREFIXES)}
    blob = json.dumps(dict(protocol=PROTOCOL, cmd=list(cmd), env=env,
                           files=[_file_sig(f) for f in sorted(set(files + [__file__]))]), sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


def _ttcalx_files(project: str) -> List[str]:
    src = Path(project) / "src"
    try:
        return [str(p) for p in sorted(src.rglob("*.jl"))]
    except OSError:
        return []


# -- server ------------------------------------------------------------------------------------------

class _Collect(logging.Handler):
    def __init__(self):
        super().__init__(logging.INFO)
        self.lines: List[str] = []

    def emit(self, record):
        try:
            self.lines.append(record.getMessage())
        except Exception:
            pass


def _send(conn: socket.socket, obj: dict) -> bool:
    try:
        conn.sendall((json.dumps(obj) + "\n").encode())
        return True
    except OSError:
        return False


def serve(spec: dict) -> int:
    """Run the server described by `spec` (keys: sock, key, julia, project, daemon_jl, sources, env,
    zest_env, start_timeout, idle_s, release_hard)."""
    from ..stages.peel import ZestDaemon
    sock_path = spec["sock"]
    idle_s = float(spec.get("idle_s", 1800.0))
    hard = bool(spec.get("release_hard", False))
    t0 = time.time()
    if spec.get("pidfile"):
        Path(spec["pidfile"]).write_text(f"{os.getpid()}\n")
    d = ZestDaemon(spec["julia"], spec["project"], spec["daemon_jl"], spec["sources"], env=spec["env"],
                   zest_env=spec["zest_env"], start_timeout=spec.get("start_timeout"))
    log.info("ZEST_SERVER key=%s daemon pid %d ready in %.1f s", spec["key"], d.pid, time.time() - t0)
    try:
        os.unlink(sock_path)
    except FileNotFoundError:
        pass
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock_path)
    os.chmod(sock_path, 0o600)
    srv.listen(4)
    srv.settimeout(idle_s)
    n_clients = n_slots = 0
    stop = False

    def release(rid_hard: bool) -> None:
        rid = next(d._ids)
        d._reply(f"RELEASE {rid}" + (" hard" if rid_hard else ""), rid, 300.0, ok_token="RELEASE_OK")

    try:
        while not stop:
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                log.info("ZEST_SERVER idle for %.0f s: exiting", idle_s)
                break
            n_clients += 1
            released = True              # a fresh daemon or one released by the previous client
            coll = _Collect()
            logging.getLogger("gsi.peel").addHandler(coll)
            try:
                rf = conn.makefile("r")
                for line in rf:
                    try:
                        req = json.loads(line)
                    except ValueError:
                        continue
                    op = req.get("op")
                    coll.lines = []
                    rep: dict = dict(ok=True)
                    try:
                        if op == "hello":
                            rep.update(key=spec["key"], pid=os.getpid(), daemon_pid=d.pid, restarts=d.restarts,
                                       clients=n_clients, slots=n_slots, protocol=PROTOCOL)
                        elif op == "zest":
                            released = False
                            rep["s"] = d.zest_slot(Path(req["slot"]), timeout=req.get("timeout"))
                            n_slots += 1
                        elif op == "reset_warm":
                            d.reset_warm(timeout=req.get("timeout", 60.0))
                        elif op == "release":
                            t = time.time()
                            release(bool(req.get("hard", hard)))
                            released = True
                            rep["s"] = time.time() - t
                        elif op == "quit":
                            stop = True
                        else:
                            rep = dict(ok=False, error=f"unknown op {op!r}")
                    except Exception as e:                     # noqa: BLE001 (sent to the client)
                        from ..stages.peel import ZestTimeout
                        rep = dict(ok=False, error=str(e), timeout=isinstance(e, ZestTimeout))
                    rep["log"] = coll.lines
                    rep["restarts"] = d.restarts
                    _send(conn, rep)
                    if stop:
                        break
            finally:
                logging.getLogger("gsi.peel").removeHandler(coll)
                try:
                    conn.close()
                except OSError:
                    pass
            if not released and not stop:     # the client went away without detaching (killed worker)
                log.warning("ZEST_SERVER client %d left without RELEASE: releasing", n_clients)
                try:
                    release(hard)
                except Exception as e:          # noqa: BLE001
                    log.error("ZEST_SERVER release after a lost client failed: %s; exiting", e)
                    break
    finally:
        try:
            srv.close()
            if os.path.exists(sock_path) and _owner_is_me(sock_path):
                os.unlink(sock_path)
        except OSError:
            pass
        d.close()
        log.info("ZEST_SERVER exit after %d client(s), %d slot(s)", n_clients, n_slots)
    return 0


def _alive(pid: int) -> bool:
    """`pid` runs (not a zombie; a zombie child of ours is reaped here)."""
    try:
        if os.waitpid(pid, os.WNOHANG)[0] == pid:
            return False
    except ChildProcessError:
        pass
    except OSError:
        return False
    try:
        with open(f"/proc/{pid}/stat") as fh:
            return fh.read().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


def _kill_stale(pidfile: Path) -> None:
    """The server named in `pidfile` (if it still runs and is a zest server of ours): SIGKILL its group."""
    try:
        pid = int(pidfile.read_text().split()[0])
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            if b"zest_server" not in fh.read():
                return
    except (OSError, ValueError, IndexError):
        return
    if not _alive(pid):
        return
    log.warning("V3_ZEST_PERSISTENT killing server pid %d (no answer)", pid)
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        return
    for _ in range(300):
        if not _alive(pid):
            return
        time.sleep(0.1)


def _owner_is_me(p: str) -> bool:
    try:
        return os.stat(p).st_uid == os.getuid()
    except OSError:
        return False


def main(argv: Optional[List[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    spec = json.loads(Path(argv[0]).read_text())
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    import faulthandler
    faulthandler.register(signal.SIGUSR1, all_threads=True)      # kill -USR1 <pid>: stacks into the log
    return serve(spec)


# -- client ------------------------------------------------------------------------------------------

class _ServerProc:
    """What the worker's budget kill needs from ``daemon._p``: kill the server and its daemon."""

    def __init__(self, pid: int):
        self.pid = pid

    def kill(self) -> None:
        try:
            os.killpg(self.pid, signal.SIGKILL)       # the server leads its own session / group
        except (ProcessLookupError, PermissionError):
            pass


class PersistentZest:
    """ZestDaemon's interface (zest_slot, reset_warm, close, pid, _p) on a per-GPU server."""

    def __init__(self, sock: socket.socket, lock_fd: int, hello: dict, io_margin_s: float, release_hard: bool):
        self._sock = sock
        self._rf = sock.makefile("r")
        self._lock_fd = lock_fd
        self.hello = hello
        self._margin = io_margin_s
        self._hard = release_hard
        self._p = _ServerProc(int(hello["pid"]))
        self.restarts = 0
        self.abandoned: List[int] = []
        self.closed = False

    @property
    def pid(self) -> int:
        return int(self.hello.get("daemon_pid", 0))

    # -- attach ----------------------------------------------------------------------------------
    @classmethod
    def attach(cls, gpu, julia: str, project: str, daemon_jl: str, sources: str, env: Dict[str, str],
               zest_env: Dict[str, str], start_timeout: Optional[float] = 600.0, idle_s: float = 1800.0,
               release_hard: bool = False, python: Optional[str] = None,
               io_margin_s: float = 120.0) -> Optional["PersistentZest"]:
        """The GPU's server (reused or started); None when another batch holds it (use a private daemon)."""
        d = base_dir()
        d.mkdir(mode=0o700, parents=True, exist_ok=True)
        tag = f"gpu{gpu}"
        lock_fd = os.open(str(d / f"{tag}.lock"), os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            os.close(lock_fd)
            if e.errno in (errno.EWOULDBLOCK, errno.EACCES):
                log.warning("V3_ZEST_PERSISTENT gpu%s: the server is attached to another batch: private daemon",
                            gpu)
                return None
            raise
        try:
            run_env = dict(env)
            run_env.update(zest_env)
            cmd = [julia, f"--project={project}", daemon_jl, sources]
            key = server_key(cmd, run_env, [daemon_jl, sources] + _ttcalx_files(project))
            sock_path = str(d / f"{tag}.sock")
            t0 = time.time()
            c = cls._connect(sock_path, timeout=30.0)
            hello = cls._hello(c) if c is not None else None
            if hello is not None and hello.get("key") == key and hello.get("protocol") == PROTOCOL:
                log.info("V3_ZEST_PERSISTENT gpu%s: reusing server pid %s (daemon pid %s, %s client(s), %s slot(s))",
                         gpu, hello["pid"], hello["daemon_pid"], hello["clients"], hello["slots"])
                return cls(c, lock_fd, hello, io_margin_s, release_hard)
            pidfile = d / f"{tag}.pid"
            if c is not None and hello is not None:
                log.warning("V3_ZEST_PERSISTENT gpu%s: server %s has another key: replacing it", gpu, hello.get("pid"))
                cls._quit(c, hello)
            elif c is not None:
                c.close()
            _kill_stale(pidfile)                     # one that does not answer (stuck, or busy with a dead client)
            spec = dict(sock=sock_path, key=key, julia=julia, project=project, daemon_jl=daemon_jl,
                        sources=sources, env=dict(env), zest_env=dict(zest_env), start_timeout=start_timeout,
                        idle_s=idle_s, release_hard=release_hard, pidfile=str(pidfile))
            spec_path = d / f"{tag}.spec.json"
            fd = os.open(str(spec_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as fh:
                json.dump(spec, fh)
            try:
                os.unlink(sock_path)
            except FileNotFoundError:
                pass
            src = str(Path(__file__).resolve().parents[2])
            senv = dict(os.environ)
            senv["PYTHONPATH"] = src + (os.pathsep + senv["PYTHONPATH"] if senv.get("PYTHONPATH") else "")
            with open(d / f"{tag}.log", "ab") as out:
                p = subprocess.Popen([python or sys.executable, "-m", "gpu_subband_imaging.v3.zest_server",
                                      str(spec_path)], stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                                     env=senv, start_new_session=True, close_fds=True)
            deadline = time.time() + (start_timeout or 3600.0) + 60.0
            while time.time() < deadline:
                if p.poll() is not None:
                    raise RuntimeError(f"zest server exited (rc={p.returncode}) before it listened; see "
                                       f"{d / (tag + '.log')}")
                c = cls._connect(sock_path, timeout=30.0) if os.path.exists(sock_path) else None
                if c is not None:
                    hello = cls._hello(c)
                    if hello is not None and hello.get("key") == key:
                        log.info("V3_ZEST_PERSISTENT gpu%s: new server pid %s (daemon pid %s) in %.1f s", gpu,
                                 hello["pid"], hello["daemon_pid"], time.time() - t0)
                        return cls(c, lock_fd, hello, io_margin_s, release_hard)
                    c.close()
                time.sleep(0.2)
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            raise RuntimeError(f"zest server not listening within {start_timeout} s (killed)")
        except BaseException:
            os.close(lock_fd)
            raise

    @staticmethod
    def _connect(path: str, timeout: float) -> Optional[socket.socket]:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(timeout)
        try:
            s.connect(path)
            return s
        except OSError:
            s.close()
            return None

    @staticmethod
    def _hello(c: socket.socket) -> Optional[dict]:
        try:
            c.sendall(b'{"op": "hello"}\n')
            line = c.makefile("r").readline()
            r = json.loads(line)
            return r if r.get("ok") else None
        except (OSError, ValueError):
            return None

    @staticmethod
    def _quit(c: socket.socket, hello: Optional[dict]) -> None:
        try:
            c.settimeout(60.0)
            c.sendall(b'{"op": "quit"}\n')
            c.makefile("r").readline()
        except OSError:
            pass
        finally:
            c.close()
        pid = (hello or {}).get("pid")
        if pid:                                     # wait for it (its daemon frees the GPU), then make sure
            for _ in range(300):
                if not _alive(int(pid)):
                    return
                time.sleep(0.1)
            try:
                os.killpg(int(pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass

    # -- requests --------------------------------------------------------------------------------
    def _request(self, wait_s: Optional[float], **req) -> dict:
        """Send `req`, return its reply; no reply within wait_s + io_margin_s (None: forever) kills the server."""
        if self.closed:
            raise RuntimeError("persistent zest client is closed")
        self._sock.settimeout(None if wait_s is None else wait_s + self._margin)
        try:
            self._sock.sendall((json.dumps(req) + "\n").encode())
            line = self._rf.readline()
        except (OSError, socket.timeout) as e:
            self._broken(f"{req.get('op')}: {e}")
            raise RuntimeError(f"zest server: no reply to {req.get('op')} ({e}); server killed") from e
        if not line:
            self._broken(f"{req.get('op')}: connection closed")
            raise RuntimeError(f"zest server closed the connection during {req.get('op')}")
        r = json.loads(line)
        for ln in r.get("log", []):
            log.info(ln)
        self.restarts = int(r.get("restarts", self.restarts))
        if not r.get("ok"):
            if r.get("timeout"):
                from ..stages.peel import ZestTimeout
                raise ZestTimeout(r.get("error", "zest timeout"))
            raise RuntimeError(r.get("error", "zest server error"))
        return r

    def _broken(self, why: str) -> None:
        """The server does not answer: kill it (with its daemon) so no stale peel touches a later slot."""
        log.error("V3_ZEST_PERSISTENT server pid %s: %s: killing it", self._p.pid, why)
        self._p.kill()
        self._drop()

    def _drop(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self._sock.close()
        except OSError:
            pass
        try:
            os.close(self._lock_fd)
        except OSError:
            pass

    def zest_slot(self, slot: Path, timeout: Optional[float] = 600.0) -> float:
        return float(self._request(timeout, op="zest", slot=str(slot), timeout=timeout)["s"])

    def reset_warm(self, timeout: Optional[float] = 60.0) -> None:
        self._request(timeout, op="reset_warm", timeout=timeout)

    def release(self, hard: Optional[bool] = None) -> float:
        h = self._hard if hard is None else hard
        return float(self._request(300.0, op="release", hard=h).get("s", 0.0))

    def quit(self) -> None:
        try:
            self._request(120.0, op="quit")
        finally:
            self._drop()

    def close(self) -> None:
        """Detach: RELEASE (the GPU memory goes back before the hour stage), keep the server."""
        if self.closed:
            return
        try:
            t = self.release()
            log.info("V3_ZEST_PERSISTENT detached from server pid %s (release %.2f s)", self._p.pid, t)
        except Exception as e:                      # noqa: BLE001 (a server that cannot release must go)
            log.error("V3_ZEST_PERSISTENT release failed (%s): killing the server", e)
            self._p.kill()
        finally:
            self._drop()


if __name__ == "__main__":
    sys.exit(main())
