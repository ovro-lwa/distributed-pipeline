"""Subband processing configuration for OVRO-LWA pipeline.

Contains node-to-subband mapping, imaging parameters, reference file paths,
and all configuration needed by the subband processing Celery tasks.

All settings live within the orca package for use with the Celery-based
subband pipeline.
"""
from astropy.coordinates import EarthLocation
import astropy.units as u

# --- Observatory Location (single source of truth) ---
OVRO_LOC = EarthLocation(lat=37.23977727*u.deg, lon=-118.2816667*u.deg, height=1222*u.m)

# --- Conda env required for running the pipeline worker ---
REQUIRED_CONDA_ENV = 'py38_orca_nkosogor'

# ---------------------------------------------------------------------------
#  Node ↔ Subband mapping
#  Each calim server has local NVMe at /fast/pipeline/ that is NOT shared.
#  Subbands are pinned to specific nodes so data stays local.
# ---------------------------------------------------------------------------
NODE_SUBBAND_MAP = {
    '18MHz': 'lwacalim00', '23MHz': 'lwacalim00',
    '27MHz': 'lwacalim01', '32MHz': 'lwacalim01',
    '36MHz': 'lwacalim02', '41MHz': 'lwacalim02',
    '46MHz': 'lwacalim03', '50MHz': 'lwacalim03',
    '55MHz': 'lwacalim04',
    '59MHz': 'lwacalim05',
    '64MHz': 'lwacalim06',
    '69MHz': 'lwacalim07',
    '73MHz': 'lwacalim08',
    '78MHz': 'lwacalim09',
    '82MHz': 'lwacalim10',
}

# Reverse map: node → list of subbands
SUBBAND_NODE_MAP = {}
for _sb, _node in NODE_SUBBAND_MAP.items():
    SUBBAND_NODE_MAP.setdefault(_node, []).append(_sb)

# All unique calim nodes
CALIM_NODES = sorted(set(NODE_SUBBAND_MAP.values()))

def get_queue_for_subband(subband: str) -> str:
    """Return the Celery queue name for a given subband.

    Args:
        subband: e.g. '73MHz'

    Returns:
        Queue name, e.g. 'calim08'
    """
    node = NODE_SUBBAND_MAP.get(subband)
    if node is None:
        raise ValueError(f"Unknown subband: {subband}")
    # Queue name = last part of hostname: lwacalim08 → calim08
    return node.replace('lwa', '')


# ---------------------------------------------------------------------------
#  Directory layout
# ---------------------------------------------------------------------------
NVME_BASE_DIR = '/fast/pipeline'           # Local NVMe on each calim node
LUSTRE_ARCHIVE_DIR = '/lustre/pipeline'    # Shared Lustre storage

# Where archive MS files live (input)
LUSTRE_NIGHTTIME_DIR = '/lustre/pipeline/night-time/averaged'

# Where final products go
LUSTRE_PRODUCTS_DIR = '/lustre/pipeline/products'


# ---------------------------------------------------------------------------
#  Reference files  (on shared Lustre, accessible from all nodes)
# ---------------------------------------------------------------------------
PEELING_PARAMS = {
    'sky_env': 'julia060',
    'rfi_env': 'ttcal_dev',
    'sky_model': '/lustre/gh/calibration/pipeline/reference/sources/sources.json',
    'rfi_model': '/lustre/gh/calibration/pipeline/reference/sources/rfi_43.2_ver20251101.json',
    'beam': 'constant',
    'minuvw': 5,
    'maxiter': 5,
    'tolerance': '1e-4',
    'args': '--beam constant --minuvw 5 --maxiter 5 --tolerance 1e-4',
}

AOFLAGGER_STRATEGY = '/lustre/ghellbourg/AOFlagger_strat_opt/LWA_opt_GH1.lua'

VLSSR_CATALOG = '/lustre/gh/calibration/pipeline/reference/surveys/FullVLSSCatalog.text'
BEAM_MODEL_H5 = '/lustre/gh/calibration/pipeline/reference/beams/OVRO-LWA_MROsoil_updatedheight.h5'


# ---------------------------------------------------------------------------
#  Hot baseline analysis parameters
# ---------------------------------------------------------------------------
HOT_BASELINE_PARAMS = {
    'run_uv_analysis': True,
    'run_heatmap_analysis': True,
    'uv_sigma': 7.0,
    'heatmap_sigma': 5.0,
    'bad_antenna_threshold': 0.25,
    'uv_cut_lambda': 4.0,
    'apply_flags': True,
}


# ---------------------------------------------------------------------------
#  Imaging configurations
#  Imaging configurations
# ---------------------------------------------------------------------------
SNAPSHOT_PARAMS = {
    'suffix': 'Pilot-Snapshot',
    'args': [
        '-log-time',
        '-pol', 'IV',
        '-niter', '0',
        '-mem', '20',
        '-size', '4096', '4096',
        '-scale', '0.03125',
        '-taper-inner-tukey', '30',
        '-weight', 'briggs', '0',
        '-no-dirty',
        '-make-psf',
        '-no-update-model-required',
    ],
}

IMAGING_STEPS = [
    # --- STOKES I ---
    {
        'pol': 'I', 'category': 'deep', 'suffix': 'I-Deep-Taper-Robust-0.75',
        'args': [
            '-log-time', '-pol', 'I', '-multiscale', '-multiscale-scale-bias', '0.8',
            '-niter', '500000', '-mgain', '0.95', '-horizon-mask', '10deg',
            '-mem', '50', '-auto-threshold', '0.5', '-auto-mask', '3', '-local-rms',
            '-size', '4096', '4096', '-scale', '0.03125',
            '-taper-inner-tukey', '30', '-weight', 'briggs', '-0.75',
            '-no-update-model-required',
        ],
    },
    {
        'pol': 'I', 'category': 'deep', 'suffix': 'I-Deep-Taper-Robust-0',
        'args': [
            '-log-time', '-pol', 'I', '-multiscale', '-multiscale-scale-bias', '0.8',
            '-niter', '500000', '-mgain', '0.95', '-horizon-mask', '10deg',
            '-mem', '50', '-auto-threshold', '0.5', '-auto-mask', '3', '-local-rms',
            '-size', '4096', '4096', '-scale', '0.03125',
            '-taper-inner-tukey', '30', '-weight', 'briggs', '0',
            '-no-update-model-required',
        ],
    },
    {
        'pol': 'I', 'category': 'deep', 'suffix': 'I-Deep-NoTaper-Robust-0.75',
        'args': [
            '-log-time', '-pol', 'I', '-multiscale', '-multiscale-scale-bias', '0.8',
            '-niter', '150000', '-mgain', '0.95', '-horizon-mask', '10deg',
            '-mem', '50', '-auto-threshold', '0.5', '-auto-mask', '3', '-local-rms',
            '-size', '4096', '4096', '-scale', '0.03125',
            '-weight', 'briggs', '-0.75',
            '-no-update-model-required',
        ],
    },
    {
        'pol': 'I', 'category': 'deep', 'suffix': 'I-Deep-NoTaper-Robust-0',
        'args': [
            '-log-time', '-pol', 'I', '-multiscale', '-multiscale-scale-bias', '0.8',
            '-niter', '150000', '-mgain', '0.95', '-horizon-mask', '10deg',
            '-mem', '50', '-auto-threshold', '0.5', '-auto-mask', '3', '-local-rms',
            '-size', '4096', '4096', '-scale', '0.03125',
            '-weight', 'briggs', '0',
            '-no-update-model-required',
        ],
    },
    {
        'pol': 'I', 'category': '10min', 'suffix': 'I-Taper-10min',
        'args': [
            '-log-time', '-pol', 'I', '-multiscale', '-multiscale-scale-bias', '0.8',
            '-niter', '50000', '-mgain', '0.95', '-horizon-mask', '10deg',
            '-mem', '50', '-auto-threshold', '0.5', '-auto-mask', '3', '-local-rms',
            '-size', '4096', '4096', '-scale', '0.03125',
            '-taper-inner-tukey', '30', '-weight', 'briggs', '0',
            '-intervals-out', '6',
            '-no-update-model-required',
        ],
    },
    # --- STOKES V ---
    {
        'pol': 'V', 'category': 'deep', 'suffix': 'V-Taper-Deep',
        'args': [
            '-log-time', '-pol', 'V', '-niter', '0',
            '-horizon-mask', '10deg', '-mem', '50',
            '-size', '4096', '4096', '-scale', '0.03125',
            '-taper-inner-tukey', '30', '-weight', 'briggs', '0',
            '-no-dirty', '-no-update-model-required',
        ],
    },
    {
        'pol': 'V', 'category': '10min', 'suffix': 'V-Taper-10min',
        'args': [
            '-log-time', '-pol', 'V', '-niter', '0',
            '-horizon-mask', '10deg', '-mem', '50',
            '-size', '4096', '4096', '-scale', '0.03125',
            '-taper-inner-tukey', '30', '-weight', 'briggs', '0',
            '-no-dirty', '-intervals-out', '6',
            '-no-update-model-required',
        ],
    },
]


# ---------------------------------------------------------------------------
#  Resource management for shared calim nodes
# ---------------------------------------------------------------------------
# Nodes with 2 subbands (e.g. calim00: 18+23 MHz) share 32 cores / 128 GB.
# Nodes with 1 subband get all resources.
_DUAL_SUBBAND_NODES = {
    n for n, subs in SUBBAND_NODE_MAP.items() if len(subs) > 1
}

def get_image_resources(subband: str):
    """Return (cpus, mem_gb, wsclean_j) for a given subband.

    On nodes that serve two subbands the resources are halved to avoid
    contention when both subbands process simultaneously.

    Args:
        subband: e.g. '73MHz'

    Returns:
        Tuple of (cpus: int, mem_gb: int, wsclean_j: int).
    """
    node = NODE_SUBBAND_MAP.get(subband)
    if node in _DUAL_SUBBAND_NODES:
        return 16, 60, 12
    return 32, 120, 24
