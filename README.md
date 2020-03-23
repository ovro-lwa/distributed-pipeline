# orca
[![Build Status](https://travis-ci.com/ovro-lwa/distributed-pipeline.svg?branch=master)](https://travis-ci.com/ovro-lwa/distributed-pipeline)
[![codecov](https://codecov.io/gh/ovro-lwa/distributed-pipeline/branch/master/graph/badge.svg)](https://codecov.io/gh/ovro-lwa/distributed-pipeline)
## Set up development environment
Before you start, it's important to make sure that other installations of casacore are not
in your `LD_LIBRARY_PATH`; otherwise it may mess up casa6.

If you have not done so, create a barebone python3.6 environment with conda.
You will need conda-forge in your channels to install pipenv. If it's not there, you can use the `-c conda-forge` flag.
```
conda create --name py36_orca python=3.6 pipenv
```

Activate with
```
conda activate py36_orca
```

Helpful environment variables to add to your `.bashrc` (and make sure these directories exist) if you're on astm:
```
export PIPENV_VENV_IN_PROJECT=1
export PIPENV_CACHE_DIR='/opt/astro/devel/<username>/cache/pipenv/'
export TMPDIR='/opt/astro/devel/<username/tmp/'
```

Run in this directory (need something about `git checkout Pipfile.lock` here?):
```
pipenv sync --dev
```
This should install the dependencies of the project (with versions etc as specified in `Pipfile.lock`). Then run the pipenv-managed virtualenv:
```pipenv shell```
or prefix any command with:
```pipenv run```

Then run
```
pipenv install -e .
```
This will install the package in development mode, so that libraries can be called and
binaries be executed. This can be re-run after code changes.

To run the tests, do
```
pytest
```
which should run the tests and output a report.

It is recommended that pycharm be used for development. I have not settled on a
style linter or a documentation format yet...

Adding a function to orca also requires integrating it with celery. This [example commit](https://github.com/ovro-lwa/distributed-pipeline/commit/e1e577437bef3c19162bdab1cd3973bee2128c04) shows the way to add and integrate a new function.

## Run with celery
TBA, but meanwhile see scripts in `proj` directory. celery admin notes are in `celery.md`.

## Code Structure
`orca` is where the wrappers and functions that do single units of work sit.

`proj` contains code that executes the pipeline with celery.
