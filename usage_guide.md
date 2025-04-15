# ORCA Usage Guide

This guide is intended to serve as a single reference point for running the core ORCA functions, as well as facilitating testing and development in a unified environment.

The shared conda environment on calim is available at:

```
conda activate /opt/devel/pipeline/envs/py38_orca_nkosogor
```

## Table of Contents
- [Concatenation](#concatenation)
- [Flagging](#flagging)
- [Model Generation](#model-generation)
- [Delay Calibration](#delay-calibration)
- [Bandpass Calibration](#bandpass-calibration)
- [Change Phase Center](#change-phase-center)
- [Imaging with WSClean](#imaging-with-wsclean)
- [Peeling with TTCal](#peeling-with-ttcal)


## Concatenation

Concatenate MS files and apply mstransform:

```python
from casatasks import concat, mstransform

concat(vis_list, outputvis, timesort=True)
mstransform(vis=outputvis, outputvis=finalvis, datacolumn='data', combinespws=True)
```

Concatenate MS files and fix field id.

```python
from casatasks import concat
from orca.utils.msfix import concat_issue_fieldid

# Concatenate MS files
concat(vis=ms_files, concatvis=output_ms, timesort=True)

# Fix FIELD_ID (and optionally OBSERVATION_ID) for compatibility
concat_issue_fieldid(output_ms, obsid=True)
```

## Flagging

Obtain bad antenna list and flag:

```python
from orca.utils.flagutils import get_bad_antenna_numbers
from orca.transform.flagging import flag_with_aoflagger, flag_ants

ants = get_bad_antenna_numbers("2025-01-28 19:20:04")
flag_ants(ms, ants)

flag_with_aoflagger(
    ms=ms, strategy="/opt/share/aoflagger/strategies/nenufar-lite.lua", 
    n_threads=5        
)

```


## Model Generation

Generate a component list.

Older version:
```python
from orca.utils.calibrationutils import gen_model_ms_stokes
gen_model_ms_stokes(ms)
```

Newer version:
```python
from orca.utils.calibratormodel import model_generation

ms_model = model_generation(ms)
ms_model.primary_beam_model = "/path/to/beam_model.h5" # path to beam model, e.g., '/lustre/ai/beam_testing/OVRO-LWA_soil_pt.h5'
cl_path, _ = ms_model.gen_model_cl(included_sources=["CygA"]) # if included_sources not provided all sources are used
```

## Delay Calibration

Clear previous model data, apply the model with `ft` and run delay calibration with `gaincal`.

```python
from casatasks import clearcal, ft, gaincal

# Clear old model and prepare MODEL_DATA column
clearcal(vis=ms, addmodel=True)

# Add component list model to data
ft(vis=ms, complist=cl_path, usescratch=True)

# Solve for delay calibration
gaincal(
    vis=ms,
    caltable=delay_table,
    uvrange='>10lambda,<125lambda',
    solint='inf',
    refant='202',
    minblperant=4,
    minsnr=3.0,
    solnorm=False,
    normtype='mean',
    gaintype='K',
    calmode='ap',
    parang=False
)
```

## Bandpass Calibration

```python
from casatasks import bandpass

bandpass(vis=ms, caltable=bandpass_table, uvrange='>10lambda,<125lambda', gaintable=delay_table, ...)
```

## Change Phase Center

Retrieve the phase center from a reference MS and apply it to another, or define one manually using RA/Dec.

```python
from orca.wrapper.change_phase_centre import get_phase_center, change_phase_center

# Get phase center from a reference MS
center_coord = get_phase_center(reference_ms)

# Convert for change_phase_center
center_str = center_coord.to_string('hmsdms')

# Apply new phase center to target MS
change_phase_center(target_ms, center_str)
```

## Imaging with WSClean

Run WSClean imaging.

```python
from orca.wrapper.wsclean import wsclean

ms_list = ['/path/to/your_measurement_set.ms']
out_dir = '/path/to/output_directory'
filename_prefix = 'image_output'

# Example WSClean arguments, equivalent to:
# /opt/bin/wsclean -pol IV -size 4096 4096 -scale 0.03125 -niter 0 -mgain 0.85 \
# -weight briggs 0 -horizon-mask 10deg -no-update-model-required ..

extra_args = [
    '-pol', 'IV',
    '-size', '4096', '4096',
    '-scale', '0.03125',
    '-niter', '0',
    '-mgain', '0.85',
    '-weight', 'briggs', '0',
    '-horizon-mask', '10deg',
    '-no-update-model-required'
]

# Run imaging
wsclean(
    ms_list=ms_list,
    out_dir=out_dir,
    filename_prefix=filename_prefix,
    extra_arg_list=extra_args,
    num_threads=4,
    mem_gb=50
)
```

## Peeling with TTCal

Run source peeling using the `peel_with_ttcal` wrapper, which internally runs the Julia-based TTCal pipeline.

```python
from orca.wrapper.ttcal import peel_with_ttcal

# Path to measurement set and associated source model (JSON)
ms = '/path/to/your_measurement_set.ms'
sources_json = '/home/pipeline/sources.json'

# Run TTCal peeling
peel_with_ttcal(ms, sources_json)
```
