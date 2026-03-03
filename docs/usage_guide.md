# ORCA Usage Guide

This guide is intended to serve as a single reference point for running the core ORCA functions, as well as facilitating testing and development in a unified environment.


## Configuration and Environment activation

If you haven’t done this already, copy the default configuration file before running any functions.

Option 1: If you want to clone the repository:

```bash
git clone https://github.com/ovro-lwa/distributed-pipeline.git
cd distributed-pipeline
cp orca/default-orca-conf.yml ~/orca-conf.yml
```

Option 2: If you don't need the full repo, then you can download the file directly:

```bash
wget https://raw.githubusercontent.com/ovro-lwa/distributed-pipeline/main/orca/default-orca-conf.yml -O ~/orca-conf.yml
```

This file includes paths and telescope-specific settings used by the pipeline. You can edit it as needed.

For more details, see the [README](../README.md#configuration-setup).


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
- [QA Plotting](#qa-plotting)



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

### Obtain and flag bad antennas (by correlator number)

```python
from orca.utils.flagutils import get_bad_correlator_numbers
from orca.transform.flagging import flag_ants

bad_corr_nums = get_bad_correlator_numbers("2025-01-28 19:20:04")
flag_ants(ms, bad_corr_nums)
```

`flag_ants()` expects a list of correlator numbers (e.g., `[41, 69, 126, ...]`) and flags the corresponding antennas in the given MeasurementSet.

If you need antenna names, use:

```python
from orca.utils.flagutils import get_bad_antenna_names

bad_ant_names = get_bad_antenna_names("2025-01-28 19:20:04")
# Example output: ['LWA-005B', 'LWA-069A', 'LWA-103A', ...]
```

To flag by antenna name instead of correlator number, use `from casatasks import flagdata`.
Note that these are `LWA-XXXA/B` names and should be parsed to proper CASA antenna names by removing the dash and polarization suffix, e.g., `LWA-005B → LWA005`. Then pass them to `flagdata` like this:

```python
flagdata(vis=ms, mode='manual', antenna='LWA005,LWA068,LWA069,...', datacolumn='all')
```


### Flag using AOFlagger and a specified strategy

```python
from orca.transform.flagging import flag_with_aoflagger
from orca.utils.paths import get_aoflagger_strategy

strategy_path = get_aoflagger_strategy("LWA_opt_GH1.lua")  # or "nenufar-lite.lua"

flag_with_aoflagger(
    ms=ms,
    strategy=strategy_path,
    n_threads=5
)
```

Strategy files are included in the repository under `orca/resources/aoflagger_strategies/` and accessed using `get_aoflagger_strategy`.


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

# If the MS filename ends with '.ms', you can create the delay table name by replacing it with '.delay'
delay_table = ms.replace('.ms', '.delay')

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
## QA Plotting


### Delay Calibration QA Plot

Generates a PDF plot of delay (in nanoseconds) per antenna or correlator number.

```python
from orca.transform.qa_plotting import plot_delay_vs_antenna

# Default: x-axis is correlator number
plot_delay_vs_antenna("your_table.delay", output_pdf="delay_vs_antenna.pdf")

# Optional: x-axis sorted by antenna number
plot_delay_vs_antenna("your_table.delay", output_pdf="delay_vs_antenna_antenna_sorted.pdf", use_antenna_labels=True)

# Optional: apply custom Y-axis limits (e.g., zoom in)
plot_delay_vs_antenna("your_table.delay", ymin=-2000, ymax=2000)
```

Also supports plotting delay **differences** between two tables:

```python
from orca.transform.qa_plotting import plot_delay_difference_vs_antenna

# Compare two delay tables (default x-axis is correlator)
plot_delay_difference_vs_antenna(
    "new_table.delay",
    reference_delay_table="reference_table.delay",
    output_pdf="delay_diff.pdf"
)

# With x-axis sorted by antenna number
plot_delay_difference_vs_antenna(
    "new_table.delay",
    reference_delay_table="reference_table.delay",
    output_pdf="delay_diff_ant.pdf",
    use_antenna_labels=True
)

# With custom Y-axis limits
plot_delay_difference_vs_antenna(
    "new_table.delay",
    reference_delay_table="reference_table.delay",
    ymin=-50,
    ymax=50
)
```

### Bandpass Calibration QA Plot

Generates a multi-page PDF showing amplitude and phase per correlator number.

```python
from orca.transform.qa_plotting import plot_bandpass_to_pdf_amp_phase

# Default (log scale with automatic limits if needed)
plot_bandpass_to_pdf_amp_phase("your_table.bandpass", pdf_path="./bandpass_QA.pdf", msfile="your_table.ms")

# Use linear scale with custom limits
plot_bandpass_to_pdf_amp_phase(
    "your_table.bandpass",
    amp_scale="linear",
    amp_limits=(1e-5, 1e-1),
    pdf_path="./bandpass_QA_linear_custom.pdf",
    msfile="your_table.ms"
)
```

Amplitude plots support `log` or `linear` scale. You can optionally provide amplitude limits. If data falls outside the given or default range, autoscaling is applied. Phase is always fixed to [-180, 180] degrees.



### Gain Calibration QA

Identify bad correlator numbers and channels from a gain calibration table using amplitude and SNR stats.

```python
from orca.transform.qa import gainQA

bad_ants, bad_ch = gainQA(
    calfile="path/to/your_table.bandpass",
    do_plot=True,                # Save AMP and SNR histograms as .png
    save_stats=True,             # Write .qa text report
    outdir="./qa_outputs",       # Output folder for plots
    qa_file="./my_table.qa"      # Optional: path to write the QA report
)
```
Saves `gains_AMP.png`, `gains_SNR.png` in the outdir (default is current dir), and a `.qa` summary in the qafile (default is calfile + '.qa').
Returns lists of bad correlator numbers and bad channels.

