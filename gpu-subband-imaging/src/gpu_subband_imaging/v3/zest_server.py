"""v3.zest_persistent: one zest daemon per GPU that outlives the batch (no Julia JIT per batch).

Each band-hour batch is its own worker process, and its first peel compiles TTCalX (~40 s on the
critical path).  With the option, the daemon runs under a small server process per GPU and node
(``python -m gpu_subband_imaging.v3.zest_server <spec.json>``, its own session) that a batch attaches
to over a UNIX socket and detaches from at the end of its snapshot loop:

* attach (``PersistentZest.attach``): per-GPU flock; a live server with the same key (the daemon's
  command, its ZEST_* / JULIA_* / CUDA_* environment and the files it loads, including the ones ZEST_*
  values name such as ZEST_RFI_SOURCES) is reused, else it is
  told to quit and a new one is started.  The flock is held while attached; a second batch on the
  same GPU (lanes, prefetch) does not wait for it but gets a private daemon (``None``).
* detach (``close``): RELEASE -- the daemon frees its workspace, device slot buffers and warm-start
  gains, collects and gives the pool back to the driver (an idle daemon then holds only its CUDA
  context, ~0.27 GB; destroying the context too crashes the next peel with TTCalX v0.5.0), then the
  connection is closed.  A client that disconnects without it (a killed worker) is released by
  the server.
* the server quits its daemon and exits after ``idle_s`` without a client, and on ``quit``.

The daemon is the ordinary ZestDaemon (timeouts, restarts, stale replies as before); its PEEL_* /
ZEST_TIMING log lines are returned with each reply and logged by the worker under gsi.peel (the server's
own log keeps only ZEST_SERVER* lines and warnings / errors).  A server lost mid-batch (no reply within
timeout + io_margin_s: killed; or it died) fails the request in flight, as a ZestDaemon restart does, and
the rest of the batch peels with a private ZestDaemon.  ``gsi cleanup-nodes`` stops the servers.

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


# the Python the server runs (this module and the ZestDaemon it drives): an in-place deploy restarts old servers
_SERVER_FILES = [__file__, str(Path(__file__).resolve().parents[1] / "stages" / "peel.py")]


def server_key(cmd: List[str], run_env: Dict[str, str], files: List[str]) -> str:
    """Two batches may share a daemon iff this is equal: the daemon command, the environment that
    reaches its behaviour, and the size / mtime of the files it loads (daemon script, sky model,
    TTCalX sources) or runs (this module, stages/peel.py)."""
    env = {k: v for k, v in run_env.items() if k.startswith(_KEY_ENV_PREFIXES)}
    blob = json.dumps(dict(protocol=PROTOCOL, cmd=list(cmd), env=env,
                           files=[_file_sig(f) for f in sorted(set(files + _SERVER_FILES))]), sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


def _env_files(run_env: Dict[str, str]) -> List[str]:
    """The files the daemon's ZEST_* settings name (ZEST_RFI_SOURCES, ...): the daemon reads them once at its start,
    so a model rewritten in place must not be served by a daemon that loaded the old one."""
    return sorted(v for k, v in run_env.items() if k.startswith("ZEST_") and v.startswith("/") and os.path.isfile(v))


def _ttcalx_files(project: str) -> List[str]:
    src = Path(project) / "src"
    try:
        files = [str(p) for p in sorted(src.rglob("*.jl"))]
    except OSError:
        files = []
    return files + [str(Path(project) / f) for f in ("Project.toml", "Manifest.toml")]


def _daemon_files(daemon_jl: str) -> List[str]:
    """The daemon script and what it includes from its own directory (zest_smooth.jl)."""
    return [daemon_jl, str(Path(daemon_jl).parent / "zest_smooth.jl")]


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
    zest_env, start_timeout, idle_s, pidfile)."""
    from ..stages.peel import ZestDaemon
    sock_path = spec["sock"]
    idle_s = float(spec.get("idle_s", 1800.0))
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

    def release() -> None:
        rid = next(d._ids)
        d._reply(f"RELEASE {rid}", rid, 300.0, ok_token="RELEASE_OK")

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
                            release()
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
                for f in (rf.close, conn.close):
                    try:
                        f()
                    except (OSError, NameError):
                        pass
            if not released and not stop:     # the client went away without detaching (killed worker)
                log.warning("ZEST_SERVER client %d left without RELEASE: releasing", n_clients)
                try:
                    release()
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


class _ServerOnly(logging.Filter):
    """The server log keeps the server's own events (ZEST_SERVER*) and every warning / error (ZEST_DAEMON_KILL,
    _RESTART, ZEST_STALE_REPLY, ...), not the daemon's per-snapshot PEEL_* / ZEST_TIMING lines."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True
        try:
            return record.getMessage().startswith("ZEST_SERVER")
        except Exception:                               # noqa: BLE001
            return True


def _setup_server_logging() -> None:
    # the daemon's PEEL_* / ZEST_TIMING lines go back to the worker with each reply (_Collect on gsi.peel); the
    # server log (/tmp/gsi_zest_<uid>/gpuN.log, appended across nights) keeps only the server's own lines
    fmt = logging.Formatter("%(asctime)s %(name)s %(message)s")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    pl = logging.getLogger("gsi.peel")
    pl.setLevel(logging.INFO)
    pl.propagate = False
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(fmt)
    h.addFilter(_ServerOnly())
    pl.addHandler(h)


def main(argv: Optional[List[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    _setup_server_logging()
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

    def __init__(self, sock: socket.socket, lock_fd: int, hello: dict, io_margin_s: float,
                 private_kw: Optional[dict] = None):
        self._sock = sock
        self._rf = sock.makefile("r")
        self._lock_fd = lock_fd
        self.hello = hello
        self._margin = io_margin_s
        self._server_p = _ServerProc(int(hello["pid"]))
        self.restarts = 0
        self.abandoned: List[int] = []
        self.closed = False                 # no server connection (detached, or the server was lost)
        self._finished = False              # close() was called: no more requests at all
        # the server lost mid-batch (killed after no reply, or it died): the rest of the batch peels with a private
        # ZestDaemon(**private_kw), as a ZestDaemon restarts its own daemon, instead of failing every snapshot left
        self._private_kw = private_kw
        self._private = None
        self._private_error: Optional[BaseException] = None

    @property
    def _p(self):
        """What the worker's budget kill kills: the server (with its daemon), or the private daemon that replaced a
        lost server -- its current process, also after one of its own restarts."""
        if self._private is not None:
            return self._private._p
        return self._server_p

    @property
    def pid(self) -> int:
        if self._private is not None:
            return self._private.pid
        return int(self.hello.get("daemon_pid", 0))

    def _fallback(self):
        if self._private_error is not None:                 # one failed start is enough (each waits start_timeout)
            raise RuntimeError(f"zest server lost and no private daemon: {self._private_error}")
        if self._private is None:
            from ..stages.peel import ZestDaemon
            log.error("V3_ZEST_PERSISTENT server pid %s lost: a private daemon for the rest of the batch",
                      self._server_p.pid)
            try:
                self._private = ZestDaemon(**self._private_kw)
            except Exception as e:                          # noqa: BLE001
                self._private_error = e
                raise
        return self._private

    def _use_private(self) -> bool:
        if self._finished:
            raise RuntimeError("persistent zest client is closed")
        return self.closed and self._private_kw is not None

    # -- attach ----------------------------------------------------------------------------------
    @classmethod
    def attach(cls, gpu, julia: str, project: str, daemon_jl: str, sources: str, env: Dict[str, str],
               zest_env: Dict[str, str], start_timeout: Optional[float] = 600.0, idle_s: float = 1800.0,
               python: Optional[str] = None,
               io_margin_s: Optional[float] = None) -> Optional["PersistentZest"]:
        """The GPU's server (reused or started); None when another batch holds it (use a private daemon).

        A request without a reply within its timeout + `io_margin_s` kills the server.  The server's ZestDaemon
        answers a timed-out peel only after it killed (kill_wait 30 s) and restarted (<= start_timeout) its
        daemon, so the default margin is start_timeout + 90 s: a shorter one kills a healthy server."""
        if io_margin_s is None:
            io_margin_s = (start_timeout if start_timeout is not None else 600.0) + 90.0
        private_kw = dict(julia=julia, project=project, daemon_jl=daemon_jl, sources=sources, env=env,
                          zest_env=zest_env, start_timeout=start_timeout)
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
            key = server_key(cmd, run_env, _daemon_files(daemon_jl) + [sources] + _env_files(run_env)
                             + _ttcalx_files(project))
            sock_path = str(d / f"{tag}.sock")
            t0 = time.time()
            c = cls._connect(sock_path, timeout=30.0)
            hello = cls._hello(c) if c is not None else None
            if hello is not None and hello.get("key") == key and hello.get("protocol") == PROTOCOL:
                log.info("V3_ZEST_PERSISTENT gpu%s: reusing server pid %s (daemon pid %s, %s client(s), %s slot(s))",
                         gpu, hello["pid"], hello["daemon_pid"], hello["clients"], hello["slots"])
                return cls(c, lock_fd, hello, io_margin_s, private_kw)
            pidfile = d / f"{tag}.pid"
            if c is not None and hello is not None:
                log.warning("V3_ZEST_PERSISTENT gpu%s: server %s has another key: replacing it", gpu, hello.get("pid"))
                cls._quit(c, hello)
            elif c is not None:
                c.close()
            _kill_stale(pidfile)                     # one that does not answer (stuck, or busy with a dead client)
            spec = dict(sock=sock_path, key=key, julia=julia, project=project, daemon_jl=daemon_jl,
                        sources=sources, env=dict(env), zest_env=dict(zest_env), start_timeout=start_timeout,
                        idle_s=idle_s, pidfile=str(pidfile))
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
                        return cls(c, lock_fd, hello, io_margin_s, private_kw)
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
            with c.makefile("r") as rf:
                line = rf.readline()
            r = json.loads(line)
            return r if r.get("ok") else None
        except (OSError, ValueError):
            return None

    @staticmethod
    def _quit(c: socket.socket, hello: Optional[dict]) -> None:
        try:
            c.settimeout(60.0)
            c.sendall(b'{"op": "quit"}\n')
            with c.makefile("r") as rf:
                rf.readline()
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

    def _broken(self, why: str, kill_wait: float = 30.0) -> None:
        """The server does not answer: kill it (with its daemon) and wait for both to exit, as ZestDaemon._kill
        waits for its daemon, so no stale peel write-back touches the slot the caller releases next."""
        log.error("V3_ZEST_PERSISTENT server pid %s: %s: killing it", self._server_p.pid, why)
        self._server_p.kill()
        pids = [p for p in (self._server_p.pid, int(self.hello.get("daemon_pid") or 0)) if p]
        deadline = time.time() + kill_wait
        while time.time() < deadline and any(_alive(p) for p in pids):
            time.sleep(0.1)
        for p in pids:
            if _alive(p):                     # e.g. stuck in a CUDA driver call (D state)
                self.abandoned.append(p)
                log.critical("ZEST_DAEMON_ABANDONED pid=%d did not exit %.0f s after SIGKILL (D state?)", p,
                             kill_wait)
        self._drop()

    def _drop(self) -> None:
        if self.closed:
            return
        self.closed = True
        # the reader made by makefile() keeps the socket's fd open: close it too, and shut the connection down,
        # so the server sees the end of this client at once (else it waits for it forever)
        for f in (lambda: self._rf.close(), lambda: self._sock.shutdown(socket.SHUT_RDWR), lambda: self._sock.close()):
            try:
                f()
            except OSError:
                pass
        try:
            os.close(self._lock_fd)
        except OSError:
            pass

    def zest_slot(self, slot: Path, timeout: Optional[float] = 600.0) -> float:
        if self._use_private():
            return self._fallback().zest_slot(slot, timeout=timeout)
        return float(self._request(timeout, op="zest", slot=str(slot), timeout=timeout)["s"])

    def reset_warm(self, timeout: Optional[float] = 60.0) -> None:
        if self._use_private():
            return self._fallback().reset_warm(timeout=timeout)
        self._request(timeout, op="reset_warm", timeout=timeout)

    def release(self) -> float:
        return float(self._request(300.0, op="release").get("s", 0.0))

    def quit(self) -> None:
        try:
            self._request(120.0, op="quit")
        finally:
            self._drop()

    def close(self) -> None:
        """Detach: RELEASE (the GPU memory goes back before the hour stage), keep the server; or quit the
        private daemon that replaced a lost server."""
        self._finished = True
        if self._private is not None:
            try:
                self._private.close()
            finally:
                self._private = None
        if self.closed:
            return
        try:
            t = self.release()
            log.info("V3_ZEST_PERSISTENT detached from server pid %s (release %.2f s)", self._server_p.pid, t)
        except Exception as e:                      # noqa: BLE001 (a server that cannot release must go)
            log.error("V3_ZEST_PERSISTENT release failed (%s): killing the server", e)
            self._server_p.kill()
        finally:
            self._drop()


if __name__ == "__main__":
    sys.exit(main())
