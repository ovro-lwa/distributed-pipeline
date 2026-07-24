"""Load and validate the three YAML configuration files."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml

DEFAULT_LUSTRE_WRITE_ROOT = "/lustre/pipeline/snapshots_gpu_pipeline"


def _allowed_lustre_write_root() -> Path:
    return Path(os.environ.get("GSI_ALLOWED_LUSTRE_WRITE_ROOT",
                               DEFAULT_LUSTRE_WRITE_ROOT))


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
    allowed_root = _allowed_lustre_write_root()
    if p.is_absolute() and p.parts[:2] == ("/", "lustre"):
        if not _under(allowed_root, p):
            raise ConfigError(
                f"{label}={p} is a Lustre write path outside "
                f"{allowed_root}"
            )


def _validate_lustre_write_roots(cluster: "ClusterConfig",
                                 pipeline: "PipelineConfig") -> None:
    """Keep configured Lustre writes inside the allowed output tree."""
    _validate_lustre_write_path("pipeline.output_root", pipeline.output_root)
    _validate_lustre_write_path("cluster.state_dir", cluster.state_dir)
    _validate_lustre_write_path("cluster.work_root", cluster.work_root)


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


@dataclass(frozen=True)
class ImagingParams:
    weight: str = "briggs"
    robust: float = 0.0
    taper_inner_tukey: float = 30.0


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
                      aoflagger_strategy=_require(cal, "aoflagger_strategy", "cal")),
        peel=peel,
        peel_overrides=peel_overrides,
        imaging=ImagingParams(**d.get("imaging", {})),
        movie=MovieParams(**d.get("movie", {})),
        do_aoflag=bool(d.get("do_aoflag", True)),
        do_badants=bool(d.get("do_badants", True)),
        env=RuntimeEnv(**env),
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
