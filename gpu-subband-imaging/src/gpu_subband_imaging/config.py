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

    def __post_init__(self) -> None:
        choices = {"solver": ("damped", "rk4"), "model": ("kernels", "exact"),
                   "uvw": ("ms", "geometric"), "rfi_mode": ("after", "joint"),
                   "device": ("gpu", "cpu")}
        for name, allowed in choices.items():
            if getattr(self, name) not in allowed:
                raise ConfigError(f"peel.{name} must be one of {allowed}, "
                                  f"got {getattr(self, name)!r}")
        if self.device == "cpu" and self.model != "exact":
            raise ConfigError("peel.device=cpu needs peel.model=exact")
        if self.maxiter < 1 or self.peeliter < 1:
            raise ConfigError("peel.maxiter and peel.peeliter must be >= 1")

    def daemon_env(self, rfi_sources: Optional[str]) -> Dict[str, str]:
        """ZEST_* environment for the zest daemon."""
        opts = {k: v for k, v in vars(self).items() if k != "rfi"}
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

    def __post_init__(self) -> None:
        if self.mode not in ("dirty", "clean"):
            raise ConfigError(f"imaging.mode must be 'dirty' or 'clean', got {self.mode!r}")


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
class CalTables:
    bp: str
    xy: str
    sources: str
    aoflagger_strategy: str
    rfi_sources: Optional[str] = None     # RFI model for peel.rfi (TTCal JSON)


@dataclass(frozen=True)
class RuntimeEnv:
    conda_env: str
    dev_env: str
    julia_bin: str
    ttcalx: str
    shared_depot: str


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

    def peel_for_band(self, band: int) -> PeelParams:
        return self.peel_overrides.get(band, self.peel)


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
                      rfi_sources=cal.get("rfi_sources")),
        peel=peel,
        peel_overrides=peel_overrides,
        imaging=ImagingParams(**d.get("imaging", {})),
        movie=MovieParams(**d.get("movie", {})),
        do_aoflag=bool(d.get("do_aoflag", True)),
        do_badants=bool(d.get("do_badants", True)),
        env=RuntimeEnv(**env),
        keep_ms_root=d.get("keep_ms_root") or None,
        hour_imaging=HourImagingParams(**d.get("hour_imaging", {})),
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
        return Config(cluster, load_subbands(d / "subbands.yaml"), pipeline)
