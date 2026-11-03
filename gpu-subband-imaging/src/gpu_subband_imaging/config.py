"""Load and validate the three YAML configuration files."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml

DEFAULT_LUSTRE_WRITE_ROOT = "/lustre/pipeline/snapshots_gpu_pipeline"


def _allowed_lustre_write_roots() -> List[Path]:
    """Allowed Lustre write roots (GSI_ALLOWED_LUSTRE_WRITE_ROOT, ':'-separated)."""
    env = os.environ.get("GSI_ALLOWED_LUSTRE_WRITE_ROOT", DEFAULT_LUSTRE_WRITE_ROOT)
    return [Path(p) for p in env.split(os.pathsep) if p]


class ConfigError(ValueError):
    pass


def _require(d: dict, key: str, where: str):
    if key not in d:
        raise ConfigError(f"missing '{key}' in {where}")
    return d[key]


def _under(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _validate_lustre_write_path(label: str, path: Union[str, Path]) -> None:
    p = Path(path)
    allowed = _allowed_lustre_write_roots()
    if p.is_absolute() and p.parts[:2] == ("/", "lustre"):
        if not any(_under(root, p) for root in allowed):
            raise ConfigError(
                f"{label}={p} is a Lustre write path outside "
                f"{os.pathsep.join(str(r) for r in allowed)}"
            )


def _validate_lustre_write_roots(cluster: "ClusterConfig",
                                 pipeline: "PipelineConfig") -> None:
    """Keep configured Lustre writes inside the allowed output tree."""
    _validate_lustre_write_path("pipeline.output_root", pipeline.output_root)
    _validate_lustre_write_path("cluster.state_dir", cluster.state_dir)
    _validate_lustre_write_path("cluster.work_root", cluster.work_root)
    if pipeline.keep_ms_root:
        _validate_lustre_write_path("pipeline.keep_ms_root", pipeline.keep_ms_root)
    if pipeline.v3.archive_root:
        _validate_lustre_write_path("pipeline.v3.archive_root", Path(pipeline.v3.archive_root).resolve())


# ---- cluster ---------------------------------------------------------------

@dataclass(frozen=True)
class NodeSpec:
    alias: str
    gpus: List[int]
    disabled_gpus: List[int] = field(default_factory=list)
    skip: bool = False

    def usable_gpus(self) -> List[int]:
        return [g for g in self.gpus if g not in self.disabled_gpus]


@dataclass(frozen=True)
class ClusterConfig:
    state_dir: str
    max_feeders: int
    wsclean_threads: int
    aoflagger_threads: int
    free_core_budget: int
    work_root: str
    nodes: List[NodeSpec]
    archive_workers: int = 1          # parallel image+archive threads per GPU slot
    zest_recycle_files: int = 25      # restart the zest daemon after this many files (0 = never)

    def slots(self) -> List["Slot"]:
        return [Slot(n.alias, g) for n in self.nodes if not n.skip
                for g in n.usable_gpus()]


@dataclass(frozen=True)
class Slot:
    node: str
    gpu: int

    def __str__(self) -> str:
        return f"{self.node}:gpu{self.gpu}"


def load_cluster(path: Union[str, Path]) -> ClusterConfig:
    d = yaml.safe_load(Path(path).read_text())
    dfl = d.get("defaults", {})
    nodes = [NodeSpec(alias=_require(n, "alias", "node"),
                      gpus=list(_require(n, "gpus", "node")),
                      disabled_gpus=list(n.get("disabled_gpus", [])),
                      skip=bool(n.get("skip", False)))
             for n in _require(d, "nodes", "cluster.yaml")]
    return ClusterConfig(
        state_dir=_require(d.get("control", {}), "state_dir", "control"),
        max_feeders=int(dfl.get("max_feeders", 3)),
        wsclean_threads=int(dfl.get("wsclean_threads", 3)),
        aoflagger_threads=int(dfl.get("aoflagger_threads", 2)),
        free_core_budget=int(dfl.get("free_core_budget", 12)),
        work_root=dfl.get("work_root", "/fast/pipeline/gpu_peel"),
        nodes=nodes,
        archive_workers=int(dfl.get("archive_workers", 1)),
        zest_recycle_files=int(dfl.get("zest_recycle_files", 25)),
    )


# ---- subbands --------------------------------------------------------------

@dataclass(frozen=True)
class ImagingGeom:
    pixels: int
    scale: float          # deg/px


@dataclass(frozen=True)
class SubbandConfig:
    geom_by_band: Dict[int, ImagingGeom]

    def geom(self, band: int) -> ImagingGeom:
        if band not in self.geom_by_band:
            raise ConfigError(f"no imaging geometry for band {band}MHz")
        return self.geom_by_band[band]


def load_subbands(path: Union[str, Path]) -> SubbandConfig:
    d = yaml.safe_load(Path(path).read_text())
    tiers = {name: ImagingGeom(int(t["pixels"]), float(t["scale"]))
             for name, t in _require(d, "tiers", "subbands.yaml").items()}
    geom = {}
    for band, tier in _require(d, "bands", "subbands.yaml").items():
        if tier not in tiers:
            raise ConfigError(f"band {band} references unknown tier '{tier}'")
        geom[int(band)] = tiers[tier]
    return SubbandConfig(geom)


# ---- pipeline --------------------------------------------------------------

@dataclass(frozen=True)
class PeelParams:
    """Zest settings, passed to the TTCalX daemon as ZEST_<FIELD> variables.

    The defaults reproduce the original GPU peeling.  The CPU-TTCal-matched
    configuration (same result as the main pipeline's ttcal.jl zest) is
    solver=rk4, maxiter=5, tolerance=1e-4, minuvw=5, min_source_elevation_deg=0,
    require_convergence=false, model=exact, uvw=geometric, flag_model=true,
    resubtract=true; rfi=true adds the RFI zest after the sky (cal.rfi_sources).
    solver=rk4, model=exact, uvw=geometric, fast_io and device=cpu need a TTCalX
    with solve_gains! (branch feature/cpu-matched-zest).
    """
    maxiter: int = 30
    tolerance: float = 1e-2
    minuvw: float = 10.0
    peeliter: int = 3
    min_source_elevation_deg: float = 15.0
    require_convergence: bool = True
    max_gain_amplitude: float = 100.0
    quality_sample_rows: int = 1024
    min_visibility_amplitude: float = 1e-12
    min_visibility_nonzero_fraction: float = 0.01
    solver: str = "damped"            # damped (gpu_stefcal!) | rk4 (CPU TTCal's solve!)
    model: str = "kernels"            # kernels (TTCalX genvis) | exact (CPU TTCal's model)
    uvw: str = "ms"                   # ms | geometric (precessed ITRF baselines)
    flag_model: bool = False          # give each model the data's flags
    resubtract: bool = False          # re-subtract a rejected source with its last accepted gains
    guard: bool = False               # undo a source update that raises the residual power
    rfi: bool = False                 # zest cal.rfi_sources (near-field RFI emitters)
    rfi_mode: str = "after"           # after (separate pass after the sky) | joint
    fast_io: bool = False             # read_ms_fast / write_ms_fast! (same results)
    device: str = "gpu"               # gpu | cpu (TTCalX on CPU threads)
    reg_max_offset_deg: float = 0.0   # > 0: Tukey phase prior (Regularized Peeling)
    reg_taper_deg: float = 1.0
    reg_lambda: float = 1.0
    flag_zero: bool = False           # flag XX == YY == 0 samples before peeling
    residual_guard: bool = False      # TTCalX guarded_zest!: keep an update only if it lowers the residual
    backstop: bool = False            # undo a whole sky/RFI pass that raises the residual
    # engine=sky_rfi: TTCalX.zest_sky_rfi! (branch feature/sponge-restore-fast-zest, v0.3.0)
    # with the options below; the settings above that it shares (solver, maxiter, tolerance,
    # minuvw, peeliter, max_gain_amplitude, min_source_elevation_deg, rfi_mode, flag_zero) apply
    engine: str = "daemon"            # daemon (the daemon's own guarded_zest! loop) | sky_rfi
    smooth_flux: float = 0.0          # Jy; > 0 ties faint directions' per-channel gains to a band gain
    # Only on the subbands where it helps (docs/peel_config.md): smooth_below_mhz > 0 applies
    # smooth_flux (and rfi_smooth_flux) only when the subband centre frequency is below it;
    # smooth_exempt_flux > 0 keeps per-channel gains for sky sources whose catalog flux at the
    # subband centre is >= it (TTCalX smooth_exempt_flux); rfi_smooth_flux (None = smooth_flux)
    # is the RFI emitters' value; smooth_exempt_below_el_deg > 0 keeps per-channel gains for sky
    # sources below that elevation in each snapshot (robust across LST; needs TTCalX with
    # zest_sky_rfi!(...; smooth_exempt_names), branch exp/sponge-exempt-apparent).
    # The defaults (0, 0, None, 0) leave smoothing as before.
    smooth_below_mhz: float = 0.0
    smooth_exempt_flux: float = 0.0
    rfi_smooth_flux: Optional[float] = None
    smooth_exempt_below_el_deg: float = 0.0
    restore: str = ""                 # point sources given back after peeling (TTCal JSON, e.g. Jupiter)
    restore_below_mhz: float = 41.0
    restore_offset_deg: float = 0.0   # Stokes V position search radius (0.5 for Jupiter)
    precision: str = "double"         # double | mixed (normal equations in single precision)
    warm_start: bool = False          # start from the previous snapshot's gains (same channels, <= 60 s)
    workspace: bool = False           # sky_rfi: reuse TTCalX PeelWorkspace buffers (same results, less GC)
    # TTCalX 0.3.0 GPU kernels (tiled normal equations, device RFI fringes, fused corrupt-subtract;
    # TTCalX.fast_kernels!): off = 748884a's kernels (bitwise), on = equal to rounding.  v3 prodlike: true
    fast_kernels: bool = False
    # comma list of passes ("sky", "sky,rfi") whose final gains the daemon writes per snapshot
    # (stages/peel_solutions.py), merged per band-hour into <band dir>/peeling_solutions/.
    # Not part of NEW_DEFAULTS: empty = no ZEST_SAVE_SOLUTIONS and no provenance key, either engine.
    save_solutions: str = ""

    def __post_init__(self) -> None:
        if any(p not in ("sky", "rfi") for p in self.solution_passes):
            raise ConfigError(f"peel.save_solutions must list sky and/or rfi, got {self.save_solutions!r}")
        choices = {"solver": ("damped", "rk4"), "model": ("kernels", "exact"),
                   "uvw": ("ms", "geometric"), "rfi_mode": ("after", "joint"),
                   "device": ("gpu", "cpu"), "engine": ("daemon", "sky_rfi"),
                   "precision": ("double", "mixed")}
        for name, allowed in choices.items():
            if getattr(self, name) not in allowed:
                raise ConfigError(f"peel.{name} must be one of {allowed}, "
                                  f"got {getattr(self, name)!r}")
        if self.device == "cpu" and self.model != "exact":
            raise ConfigError("peel.device=cpu needs peel.model=exact")
        if self.workspace and self.engine != "sky_rfi":
            raise ConfigError("peel.workspace needs peel.engine=sky_rfi")
        if self.maxiter < 1 or self.peeliter < 1:
            raise ConfigError("peel.maxiter and peel.peeliter must be >= 1")
        for name in ("smooth_flux", "smooth_below_mhz", "smooth_exempt_flux", "rfi_smooth_flux",
                     "smooth_exempt_below_el_deg"):
            value = getattr(self, name)
            if value is None and name == "rfi_smooth_flux":
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not value >= 0:
                raise ConfigError(f"peel.{name} must be >= 0, got {value!r}")

    # The fields added after 13405ce (engine .. fast_kernels) with their defaults.  While every one of them
    # is at its default, neither the ZEST_* environment nor the provenance "peel" block has them, so
    # both are exactly 13405ce's; once one is set, all of them are given (as the sky_rfi tests ran).
    NEW_DEFAULTS = {"engine": "daemon", "smooth_flux": 0.0, "smooth_below_mhz": 0.0,
                    "smooth_exempt_flux": 0.0, "rfi_smooth_flux": None,
                    "smooth_exempt_below_el_deg": 0.0, "restore": "", "restore_below_mhz": 41.0,
                    "restore_offset_deg": 0.0, "precision": "double", "warm_start": False,
                    "workspace": False, "fast_kernels": False}
    # with the new fields in use: these are still never sent at their defaults
    _UNSET = {"smooth_below_mhz": 0.0, "smooth_exempt_flux": 0.0, "rfi_smooth_flux": None,
              "smooth_exempt_below_el_deg": 0.0, "workspace": False, "fast_kernels": False,
              "save_solutions": ""}

    @property
    def solution_passes(self) -> List[str]:
        return [p.strip() for p in self.save_solutions.split(",") if p.strip()]

    def new_fields_default(self) -> bool:
        return all(getattr(self, k) == v for k, v in self.NEW_DEFAULTS.items())

    def as_dict(self) -> Dict[str, object]:
        """The settings for provenance.json: 13405ce's keys while the new fields are at their defaults."""
        d = dict(vars(self))
        if self.new_fields_default():
            for k in self.NEW_DEFAULTS:
                d.pop(k, None)
        if not self.save_solutions:
            d.pop("save_solutions", None)
        return d

    def daemon_env(self, rfi_sources: Optional[str]) -> Dict[str, str]:
        """ZEST_* environment for the zest daemon."""
        legacy = self.new_fields_default()
        opts = {k: v for k, v in vars(self).items()
                if k != "rfi" and not (legacy and k in self.NEW_DEFAULTS)
                and not (k in self._UNSET and v == self._UNSET[k])}
        if self.rfi:
            if not rfi_sources:
                raise ConfigError("peel.rfi is true but cal.rfi_sources is not set")
            opts["rfi_sources"] = rfi_sources
        return {f"ZEST_{k.upper()}": str(v) for k, v in opts.items()}


@dataclass(frozen=True)
class ImagingParams:
    """Snapshot imaging.

    mode=dirty: one WSClean run, dirty Stokes I and V (``-pol IV -niter 0``).
    mode=clean: Stokes I CLEAN with the main pipeline's SNAPSHOT_CLEAN_I_PARAMS
    (clean_* fields; -size/-scale from the band tier) plus a dirty Stokes V run
    (the main pipeline's pilot SNAPSHOT_PARAMS with -pol V).  Both modes write
    <prefix>-I-image.fits and <prefix>-V-image.fits.
    """
    weight: str = "briggs"
    robust: float = 0.0
    taper_inner_tukey: float = 30.0
    mode: str = "dirty"                 # dirty | clean
    clean_niter: int = 50000
    clean_mgain: float = 0.9999
    clean_auto_mask: float = 5.0
    clean_auto_threshold: float = 1.0
    clean_local_rms: bool = True
    clean_horizon_mask: str = "10deg"
    clean_mem: int = 50
    dirty_mem: int = 20
    abs_mem: Optional[float] = None     # WSClean -abs-mem [GB] (caps -mem on busy nodes)
    retries: int = 0                    # re-run a failed WSClean call this many times
    # wsclean: CPU WSClean runs; ttcalx: the same CLEAN I + dirty V on the worker's GPU
    # (stages/gpu_image.py, needs mode=clean) with the ttcalx_imager venv's python
    # (default: beside hour_imaging.ttcalx_image)
    imager: str = "wsclean"
    gpu_python: Optional[str] = None
    gpu_env: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in ("dirty", "clean"):
            raise ConfigError(f"imaging.mode must be 'dirty' or 'clean', got {self.mode!r}")
        if self.imager not in ("wsclean", "ttcalx"):
            raise ConfigError(f"imaging.imager must be 'wsclean' or 'ttcalx', got {self.imager!r}")
        if self.imager == "ttcalx" and self.mode != "clean":
            raise ConfigError("imaging.imager=ttcalx needs imaging.mode=clean")


@dataclass(frozen=True)
class HourImagingParams:
    """1-h / 10-min products from the kept peeled MSes with ``ttcalx-image hour``.

    products are main-pipeline names (I-Deep-Taper-Robust-0, I-Deep-Taper-Robust-0.75,
    I-Deep-NoTaper-Robust-0.75, I-Deep-NoTaper-Robust-0, I-Taper-10min; V-Taper-Deep and
    V-Taper-10min come with the Robust-0 tapered products).
    """
    ttcalx_image: str = "ttcalx-image"
    products: List[str] = field(default_factory=lambda: [
        "I-Deep-Taper-Robust-0", "I-Deep-Taper-Robust-0.75",
        "I-Deep-NoTaper-Robust-0.75", "I-Deep-NoTaper-Robust-0",
        "I-Taper-10min", "V-Taper-Deep", "V-Taper-10min"])
    nproc: int = 8                      # MS reader processes
    qa: str = "compute"                 # ttcalx-image --qa
    extra_args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    timeout_seconds: Optional[int] = None
    delete_kept_ms: bool = False        # remove a band-hour's kept MSes after it imaged successfully
    # false: the snapshots the pilot Stokes V QA rejects stay in the hour products (the QA table
    # is still written); true (default) leaves them out
    qa_exclude: bool = True


@dataclass(frozen=True)
class MovieParams:
    enabled: bool = True
    max_px: int = 512
    fps: int = 20
    crf: int = 28
    scale_mode: str = "adaptive_i"
    fixed_vmax: float = 20.0
    rms_scale: float = 10.0
    horizon_mask: bool = True
    horizon_radius_fraction: float = 0.49
    keep_frames: bool = True          # if False, delete PNG frames after stitch


@dataclass(frozen=True)
class V3Params:
    """gsi v3 (in-memory band-hour pipeline, design: docs/v3_design.md).

    Every default reproduces v2; each switch turns on one step of the plan.
    """
    untar_to: str = ""                # "" = work_root as v2; e.g. /dev/shm/gsi3 (tar streamed from Lustre)
    prep_inmem: str = "off"           # off | shadow (compute both, log diffs, use v2) | on
    peel_io: str = "ms"               # ms | shm
    snapshot_io: str = "ms"           # ms | shm
    hour_source: str = "kept_ms"      # kept_ms | archive | memory
    # keep writing kept MSes (the v2 hour job, comparisons); None = True except with hour_source=memory
    # (the batch makes the hour from its records: kept MSes would be ~39 GB written and deleted per band-hour)
    write_kept_ms: Optional[bool] = None
    archive: str = "none"             # none | iv | full
    archive_root: str = ""            # Lustre directory for the per-band-hour archive files
    archive_python: str = ""          # imager-venv python of the archive writer (default: imaging.gpu_python,
    #                                   else the python beside hour_imaging.ttcalx_image)
    store: str = "auto"               # ram | spill | auto
    spill_dirs: List[str] = field(default_factory=lambda: ["/fast/pipeline/gsi3_spill", "/tmp/gsi3_spill"])
    ram_reserve_gb: float = 16.0
    checkpoint: str = "local"         # none | local | archive (only when hour_source != kept_ms)
    delete_checkpoint: bool = True
    lookahead: int = 8
    peel_timeout_s: float = 600.0     # peel_io=shm: kill + restart the zest daemon after this
    peel_start_timeout_s: float = 600.0   # peel_io=shm: deadline for the daemon's DAEMON_READY
    prep_procs: int = 4
    prep_timeout_s: float = 900.0     # per in-memory prep task (as the v2 CASA subprocess)
    peel_backend: str = "daemon"      # daemon | juliacall
    # snapshot_io=shm: workers/gsi_v3_engine.py (imaging.gpu_python), v3/engine_client.py
    engine_timeout_s: float = 900.0   # per snapshot (image + record): kill + restart the engine after this
    engine_start_timeout_s: float = 600.0
    engine_max_restarts: int = 5      # engine restarts per batch; then the batch is aborted
    mem_check_every: int = 2          # MemAvailable checked every N snapshots (spill, then less lookahead / no tmpfs)
    # snapshot_io=shm: true = the engine images snapshot i while the daemon peels i+1 (both on the GPU,
    # time-sliced); false = the next peel waits until the previous snapshot is imaged (no GPU overlap)
    image_overlap: bool = True
    # the zest daemon's JULIA_CUDA_SOFT_MEMORY_LIMIT: "" = as v2 (4GiB when the GPU imager shares the GPU),
    # "none" = no limit, else the value (e.g. "8GiB").  With the shm hand-off there is no MS I/O between
    # peels for Julia's GC to run in, and the 4 GiB limit forces a GC inside each peel (+0.3 s at 23 MHz)
    zest_soft_memory_limit: str = ""
    zest_hard_memory_limit: str = ""  # the daemon's JULIA_CUDA_HARD_MEMORY_LIMIT ("" = unset, as v2)
    zest_gc_between: str = ""         # "" (v2) | young | full: Julia GC in the daemon after each snapshot's reply
    profile_no_image: bool = False    # profiling only: peel + record without the engine's imaging (no FITS, no hour)
    budget_margin_s: float = 300.0    # hour_source=memory: the worker stops itself this long before batch_timeout_seconds
    # hour_source=memory: an input that failed in this many batch attempts (0 = max_retries + 1) is given
    # up, and the hour is made from the rest (V3_HOUR_REDUCED, provenance hour.failed_inputs); a tar that
    # vanished is given up at once
    max_input_failures: int = 0
    # hour_source=memory: no hour products when more than this fraction of the band-hour's inputs (no-data
    # skips not counted) is given up (V3_TOO_MANY_GIVEN_UP; 1 = no cap)
    max_given_up_fraction: float = 0.1
    # hour_source=memory: local checkpoints in spill_dirs not modified for this long (a run that failed for
    # good and was never resumed) are deleted at the next batch start on the node (0 = never)
    stale_checkpoint_hours: float = 48.0
    # copy + extract attempts per tar on the v2 staging path (None: 1 as v2, 3 in v3 mode; the untar_to
    # stream path always makes 3)
    stage_attempts: Optional[int] = None

    @property
    def active(self) -> bool:
        """Any v3 step switched on ("v3 mode").  The v3-only extras follow it: QA row times, the per-row QA
        append, newest-row QA semantics, the v3 worker environment, the staging retry."""
        return bool(self.untar_to or self.prep_inmem != "off" or self.peel_io != "ms" or
                    self.snapshot_io != "ms" or self.hour_source != "kept_ms" or self.archive != "none")

    def __post_init__(self) -> None:
        if self.stage_attempts is None:
            object.__setattr__(self, "stage_attempts", 3 if self.active else 1)
        if self.write_kept_ms is None:
            object.__setattr__(self, "write_kept_ms", self.hour_source != "memory")
        for name, allowed in (("prep_inmem", ("off", "shadow", "on")), ("peel_io", ("ms", "shm")),
                              ("snapshot_io", ("ms", "shm")),
                              ("hour_source", ("kept_ms", "archive", "memory")),
                              ("archive", ("none", "iv", "full")), ("store", ("ram", "spill", "auto")),
                              ("checkpoint", ("none", "local", "archive")),
                              ("peel_backend", ("daemon", "juliacall"))):
            if getattr(self, name) not in allowed:
                raise ConfigError(f"v3.{name} must be one of {allowed}, got {getattr(self, name)!r}")
        if self.untar_to and Path(self.untar_to).parts[:2] == ("/", "lustre"):
            raise ConfigError("v3.untar_to must be node-local (tmpfs or NVMe), not Lustre")
        if self.peel_io == "shm" and self.prep_inmem != "on":
            raise ConfigError("v3.peel_io=shm needs v3.prep_inmem=on")
        if self.snapshot_io == "shm" and self.peel_io != "shm":
            raise ConfigError("v3.snapshot_io=shm needs v3.peel_io=shm")
        if self.hour_source == "memory" and self.snapshot_io != "shm":
            raise ConfigError("v3.hour_source=memory needs v3.snapshot_io=shm")
        if self.archive != "none" and not self.archive_root:
            raise ConfigError("v3.archive needs v3.archive_root")
        if self.archive_root:          # absolute, so the Lustre write check sees the real path
            object.__setattr__(self, "archive_root", os.path.abspath(self.archive_root))
        if self.archive == "full":
            raise ConfigError("v3.archive=full is not implemented yet (use iv)")
        if self.hour_source == "archive" and self.archive == "none":
            raise ConfigError("v3.hour_source=archive needs v3.archive (iv)")
        if self.hour_source == "kept_ms" and not self.write_kept_ms:
            raise ConfigError("v3.hour_source=kept_ms needs v3.write_kept_ms")
        if self.hour_source == "archive" and not self.write_kept_ms and self.snapshot_io != "shm":
            # the v2 worker moves a snapshot's FITS (the snapshot is done) before its archive record is
            # written and has no replay: a lost record would leave neither a record nor a kept MS to image
            raise ConfigError("v3.hour_source=archive with v3.write_kept_ms=false needs v3.snapshot_io=shm "
                              "(the v2 worker cannot replay a lost archive record)")
        if self.lookahead < 1 or self.prep_procs < 1:
            raise ConfigError("v3.lookahead and v3.prep_procs must be >= 1")
        if not self.prep_timeout_s > 0:
            raise ConfigError("v3.prep_timeout_s must be > 0")
        if self.peel_timeout_s <= 0 or self.peel_start_timeout_s <= 0:
            raise ConfigError("v3.peel_timeout_s and v3.peel_start_timeout_s must be > 0")
        if self.engine_timeout_s <= 0 or self.engine_start_timeout_s <= 0:
            raise ConfigError("v3.engine_timeout_s and v3.engine_start_timeout_s must be > 0")
        if self.hour_source == "memory" and self.checkpoint == "none":
            raise ConfigError("v3.hour_source=memory needs v3.checkpoint (local | archive): without a record file "
                              "an engine restart loses the band-hour's records")
        if self.zest_gc_between not in ("", "young", "full"):
            raise ConfigError("v3.zest_gc_between must be '', 'young' or 'full'")
        if self.max_input_failures < 0:
            raise ConfigError("v3.max_input_failures must be >= 0 (0 = max_retries + 1)")
        if self.engine_max_restarts < 0 or self.mem_check_every < 1 or self.budget_margin_s < 0:
            raise ConfigError("v3.engine_max_restarts >= 0, v3.mem_check_every >= 1, v3.budget_margin_s >= 0")
        if self.checkpoint == "archive" and self.archive == "none":
            raise ConfigError("v3.checkpoint=archive needs v3.archive (iv)")
        if self.stage_attempts < 1:
            raise ConfigError("v3.stage_attempts must be >= 1")
        if not 0 <= self.max_given_up_fraction <= 1 or self.stale_checkpoint_hours < 0:
            raise ConfigError("v3.max_given_up_fraction must be in [0, 1] and v3.stale_checkpoint_hours >= 0")
        for d in self.spill_dirs:
            if Path(d).parts[:2] == ("/", "lustre"):
                raise ConfigError("v3.spill_dirs must be node-local (the archive is the Lustre copy)")


def v3_params(pipeline) -> V3Params:
    """pipeline.v3, or the v2 defaults for a pipeline object without it (tests, old pickles)."""
    v = getattr(pipeline, "v3", None)
    return v if v is not None else V3Params()


@dataclass(frozen=True)
class CalTables:
    bp: str
    xy: str
    sources: str
    aoflagger_strategy: str
    rfi_sources: Optional[str] = None     # RFI model for peel.rfi (TTCal JSON)
    # fixed bad correlator numbers (CSV, {band: CSV} or {"<LST>h/<band>": CSV}) instead of the
    # live MNC anthealth lookup, which answers with the current list (backlog re-runs use the
    # list the original run flagged); as feature/gsi-streaming
    badants: Optional[Union[str, Dict[Union[int, str], str]]] = None

    def badants_for(self, band: int, group: Optional[str] = None) -> Optional[str]:
        if isinstance(self.badants, dict):
            if group is not None and f"{group}/{band}" in self.badants:
                return self.badants[f"{group}/{band}"]
            return self.badants.get(band)
        return self.badants


@dataclass(frozen=True)
class RuntimeEnv:
    conda_env: str
    dev_env: str
    julia_bin: str
    ttcalx: str
    shared_depot: str


FIX_UVW_MODES = ("off", "centroid", "time")

# Scaife & Heald (2012) flux-scale calibrators: never peeled.  A sky or RFI model with a source
# of that name, or within PROTECTED_RADIUS_DEG of one, is refused when the config loads.
PROTECTED_SOURCES = {"3C 48": ("01h37m41.3s", "+33d09m35s"), "3C 147": ("05h42m36.1s", "+49d51m07s"),
                     "3C 196": ("08h13m36.0s", "+48d13m03s"), "3C 286": ("13h31m08.3s", "+30d30m33s"),
                     "3C 295": ("14h11m20.5s", "+52d12m10s"), "3C 380": ("18h29m31.8s", "+48d44m46s")}
PROTECTED_RADIUS_DEG = 1.0


def _model_entries(path: Optional[str]) -> List[dict]:
    import json
    if not path or not Path(path).is_file():
        return []
    d = json.loads(Path(path).read_text())
    return d if isinstance(d, list) else d.get("sources", [])


def model_source_names(path: Optional[str]) -> List[str]:
    """Source names of a TTCal JSON model (empty if the file is missing)."""
    return [str(s.get("name", "?")) for s in _model_entries(path)]


def check_protected(path: Optional[str]) -> None:
    """ConfigError if the model would peel a Scaife & Heald calibrator (by name or position)."""
    entries = _model_entries(path)
    if not entries:
        return
    norm = lambda n: str(n).replace(" ", "").upper()      # noqa: E731
    names = {norm(k) for k in PROTECTED_SOURCES}
    coords = []
    for s in entries:
        if norm(s.get("name", "")) in names:
            raise ConfigError(f"{path}: peels {s['name']}, a Scaife & Heald calibrator")
        for c in [s] + list(s.get("components", []) or []):
            if c.get("ra") and c.get("dec"):
                coords.append((s.get("name", "?"), c["ra"], c["dec"]))
    if not coords:
        return
    from astropy.coordinates import SkyCoord
    prot = SkyCoord([v[0] for v in PROTECTED_SOURCES.values()], [v[1] for v in PROTECTED_SOURCES.values()])
    for name, ra, dec in coords:
        try:
            c = SkyCoord(ra, dec)
        except Exception:
            continue                       # TTCalX-only formats; the name check still applies
        sep = c.separation(prot).deg
        if sep.min() < PROTECTED_RADIUS_DEG:
            raise ConfigError(f"{path}: {name} is {sep.min():.2f} deg from "
                              f"{list(PROTECTED_SOURCES)[int(sep.argmin())]} (Scaife & Heald calibrator)")


def _fix_uvw_mode(v) -> str:
    # YAML reads a bare `off` as False.
    return "off" if v is False or v is None else str(v)


@dataclass(frozen=True)
class PipelineConfig:
    date: str
    hours: Union[str, List[str]]
    bands: List[int]
    batch_size: int
    batch_timeout_seconds: Optional[int]
    max_retries: int
    data_root: str
    output_root: str
    chanbin: int
    cal: CalTables
    peel: PeelParams
    peel_overrides: Dict[int, PeelParams]
    imaging: ImagingParams
    movie: MovieParams
    do_aoflag: bool
    do_badants: bool
    env: RuntimeEnv
    keep_ms_root: Optional[str] = None  # move peeled MSes here instead of deleting them
    hour_imaging: HourImagingParams = field(default_factory=HourImagingParams)
    # Recompute UVW in J2000 after averaging (stages/uvw.py): the raw MS (u, v) are
    # rotated ~0.34 deg about w, which rotates every snapshot image.  centroid: at
    # TIME_CENTROID (exact phase centre); time: at TIME like chgcentre (5 s early,
    # matches the main pipeline); off: keep the stored UVW.
    fix_uvw: str = "centroid"
    # layout=utc: <output_root>/<band>MHz/<date>/<UTC hour>/{fits,images} (original).
    # layout=lst: main-pipeline layout <output_root>/<LST>h/<date>/<run_label>/<band>MHz/
    # (lst.py); only the LST hours in lst_hours, with the main pipeline's Sun cut.
    layout: str = "utc"
    lst_hours: List[int] = field(default_factory=list)
    run_label: str = ""
    sun_cut: bool = True
    # zenith: image each snapshot at its own phase centre; lst: chgcentre a copy to the
    # LST-hour centre <LST>h30m00s 37d12m57.057s first (as the main pipeline's snapshots)
    snapshot_phase_centre: str = "zenith"
    chgcentre: str = "/opt/bin/chgcentre"
    # Node-local (NVMe) keep root: peeled MSes stay on the node that made them and the
    # band-hour's hour job runs there (needs stream_hour_images).  Below
    # keep_ms_local_min_free_gb free a snapshot falls back to keep_ms_root.
    keep_ms_local: Optional[str] = None
    keep_ms_local_min_free_gb: float = 100.0
    # layout=lst: start each band-hour's hour job as soon as its batches are done instead
    # of after every batch of the run; wideband_per_hour runs pipeline/wideband_summary.py
    # for an LST hour once all its band-hours are imaged.
    stream_hour_images: bool = False
    wideband_per_hour: bool = False
    # run: remove the node calibration caches at the end of each run; none: keep them
    # (scripts/run_nightly_lst.py removes them with `gsi cleanup-nodes` after the night).
    calcache_cleanup: str = "run"
    # casa: flagdata + applycal + mstransform; numpy: the same in one casacore pass
    # (stages/fastcal.py), without writing CORRECTED_DATA.
    prep: str = "casa"
    max_files_per_hour: Optional[int] = None    # testing: first N snapshots of each hour
    description: str = ""               # free text for provenance.json (what this run is); "" = no key
    v3: V3Params = field(default_factory=V3Params)

    def __post_init__(self) -> None:
        if self.chanbin < 1:
            raise ConfigError("chanbin must be >= 1 (1 = inputs already averaged)")
        if self.chanbin == 1 and self.do_aoflag:
            raise ConfigError("chanbin 1 (no averaging) needs do_aoflag false")
        peels = [self.peel, *(self.peel_overrides or {}).values()]
        if any(getattr(p, "solution_passes", None) for p in peels) and self.layout != "lst":
            raise ConfigError("peel.save_solutions needs layout=lst (merged per band-hour by the hour step)")
        if self.fix_uvw not in FIX_UVW_MODES:
            raise ConfigError(f"fix_uvw must be one of {FIX_UVW_MODES}, got {self.fix_uvw!r}")
        if self.layout not in ("utc", "lst"):
            raise ConfigError(f"layout must be 'utc' or 'lst', got {self.layout!r}")
        if self.snapshot_phase_centre not in ("zenith", "lst"):
            raise ConfigError("snapshot_phase_centre must be 'zenith' or 'lst'")
        if self.layout == "lst" and not self.run_label:
            raise ConfigError("layout=lst needs run_label (e.g. Run_20261026_031500)")
        if self.snapshot_phase_centre == "lst" and self.layout != "lst":
            raise ConfigError("snapshot_phase_centre=lst needs layout=lst")
        if self.stream_hour_images and self.layout != "lst":
            raise ConfigError("stream_hour_images needs layout=lst")
        if self.keep_ms_local:
            if not self.stream_hour_images:
                raise ConfigError("keep_ms_local needs stream_hour_images (hour jobs on the node)")
            if Path(self.keep_ms_local).parts[:2] == ("/", "lustre"):
                raise ConfigError("keep_ms_local must be node-local storage, not Lustre")
        if self.calcache_cleanup not in ("run", "none"):
            raise ConfigError("calcache_cleanup must be 'run' or 'none'")
        if self.prep not in ("casa", "numpy"):
            raise ConfigError("prep must be 'casa' or 'numpy'")
        if self.v3.prep_inmem != "off" and self.prep != "numpy":
            raise ConfigError("v3.prep_inmem needs prep=numpy")
        if self.v3.prep_inmem != "off" and self.do_aoflag:
            raise ConfigError("v3.prep_inmem needs do_aoflag: false (the numpy prep path)")
        if self.v3.snapshot_io == "shm" and self.imaging.imager != "ttcalx":
            raise ConfigError("v3.snapshot_io=shm needs imaging.imager=ttcalx")
        if self.v3.hour_source == "memory" and not (self.layout == "lst" and self.stream_hour_images):
            raise ConfigError("v3.hour_source=memory needs layout=lst and stream_hour_images")
        if self.v3.hour_source == "archive" and self.layout != "lst":
            raise ConfigError("v3.hour_source=archive needs layout=lst (hour_lst.run_band)")
        if self.v3.hour_source == "memory" and self.batch_timeout_seconds and \
                self.v3.budget_margin_s >= self.batch_timeout_seconds:
            raise ConfigError("v3.budget_margin_s must be smaller than batch_timeout_seconds")
        if self.v3.profile_no_image and not os.environ.get("GSI_V3_ALLOW_PROFILE"):
            # profiling only: the batch then always fails (no FITS, no hour), so it can never look done
            raise ConfigError("v3.profile_no_image is a profiling switch: set GSI_V3_ALLOW_PROFILE=1 to use it")
        if self.v3.hour_source == "memory" and self.batch_timeout_seconds and \
                self.hour_imaging.timeout_seconds and \
                self.batch_timeout_seconds < self.hour_imaging.timeout_seconds + 1800 + self.v3.budget_margin_s:
            raise ConfigError(
                f"v3.hour_source=memory: the batch makes the hour products, so batch_timeout_seconds "
                f"({self.batch_timeout_seconds}) must cover the snapshots (>= 1800 s), the hour "
                f"(hour_imaging.timeout_seconds {self.hour_imaging.timeout_seconds}) and v3.budget_margin_s")
        if self.v3.snapshot_io == "shm" and self.v3.archive != "none" and self.layout != "lst":
            raise ConfigError("v3.snapshot_io=shm with v3.archive needs layout=lst (one record file per batch)")

    @property
    def run_key(self) -> str:
        """Ledger/manifest key: the date, plus the LST hours for layout=lst (one run per set)."""
        if self.layout != "lst":
            return self.date
        key = f"{self.date}_{self.run_label}_lst" + "-".join(f"{h:02d}" for h in self.lst_hours)
        # hour_source=memory has another batch layout (one batch per band-hour, no hour-job rows): its own
        # ledger, so a v2 ledger of the same run is never reused (INSERT OR IGNORE on (band, batch))
        return key + "_v3mem" if self.v3.hour_source == "memory" else key

    def peel_for_band(self, band: int) -> PeelParams:
        return self.peel_overrides.get(band, self.peel)


def _badants(v):
    if v is None:
        return None
    if isinstance(v, dict):
        return {(int(k) if str(k).isdigit() else str(k)): str(csv) for k, csv in v.items()}
    return str(v)


def load_pipeline(path: Union[str, Path]) -> PipelineConfig:
    d = yaml.safe_load(Path(path).read_text())
    cal = _require(d, "cal", "pipeline.yaml")
    env = _require(d, "env", "pipeline.yaml")
    peel = PeelParams(**d.get("peel", {}))
    peel_overrides = {
        int(band): PeelParams(**{**vars(peel), **params})
        for band, params in d.get("peel_overrides", {}).items()
    }
    return PipelineConfig(
        date=str(_require(d, "date", "pipeline.yaml")),
        hours=d.get("hours", "all"),
        bands=[int(b) for b in _require(d, "bands", "pipeline.yaml")],
        batch_size=int(d.get("batch_size", 50)),
        batch_timeout_seconds=(int(d["batch_timeout_seconds"])
                               if d.get("batch_timeout_seconds") else None),
        max_retries=int(d.get("max_retries", 1)),
        data_root=_require(d, "data_root", "pipeline.yaml"),
        output_root=_require(d, "output_root", "pipeline.yaml"),
        chanbin=int(d.get("chanbin", 4)),
        cal=CalTables(bp=_require(cal, "bp", "cal"), xy=_require(cal, "xy", "cal"),
                      sources=_require(cal, "sources", "cal"),
                      aoflagger_strategy=_require(cal, "aoflagger_strategy", "cal"),
                      rfi_sources=cal.get("rfi_sources"),
                      badants=_badants(cal.get("badants"))),
        peel=peel,
        peel_overrides=peel_overrides,
        imaging=ImagingParams(**d.get("imaging", {})),
        movie=MovieParams(**d.get("movie", {})),
        do_aoflag=bool(d.get("do_aoflag", True)),
        do_badants=bool(d.get("do_badants", True)),
        env=RuntimeEnv(**env),
        keep_ms_root=d.get("keep_ms_root") or None,
        hour_imaging=HourImagingParams(**d.get("hour_imaging", {})),
        fix_uvw=_fix_uvw_mode(d.get("fix_uvw", "centroid")),
        layout=str(d.get("layout", "utc")),
        lst_hours=[int(h) for h in d.get("lst_hours", [])],
        run_label=str(d.get("run_label", "")),
        sun_cut=bool(d.get("sun_cut", True)),
        snapshot_phase_centre=str(d.get("snapshot_phase_centre", "zenith")),
        chgcentre=str(d.get("chgcentre", "/opt/bin/chgcentre")),
        keep_ms_local=d.get("keep_ms_local") or None,
        keep_ms_local_min_free_gb=float(d.get("keep_ms_local_min_free_gb", 100.0)),
        stream_hour_images=bool(d.get("stream_hour_images", False)),
        wideband_per_hour=bool(d.get("wideband_per_hour", False)),
        calcache_cleanup=str(d.get("calcache_cleanup", "run")),
        prep=str(d.get("prep", "casa")),
        max_files_per_hour=(int(d["max_files_per_hour"])
                            if d.get("max_files_per_hour") else None),
        description=str(d.get("description") or ""),
        v3=V3Params(**(d.get("v3") or {})),
    )


@dataclass(frozen=True)
class Config:
    cluster: ClusterConfig
    subbands: SubbandConfig
    pipeline: PipelineConfig

    @staticmethod
    def load(config_dir: Union[str, Path]) -> "Config":
        d = Path(config_dir)
        cluster = load_cluster(d / "cluster.yaml")
        pipeline = load_pipeline(d / "pipeline.yaml")
        _validate_lustre_write_roots(cluster, pipeline)
        check_protected(pipeline.cal.sources)
        check_protected(pipeline.cal.rfi_sources)
        return Config(cluster, load_subbands(d / "subbands.yaml"), pipeline)
