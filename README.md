# astm-transient-stuff
## Set up development environment
Run in this directory (with pipenv properly installed)
```
pipenv update --dev
```
`orca` is where the wrappers and functions that do single units of work sit.

`proj` contains code that executes the pipeline with `celery`.

`notebook` contains ad-hoc analyses and experimental things.