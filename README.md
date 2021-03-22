# orca
[![Build Status](https://travis-ci.com/ovro-lwa/distributed-pipeline.svg?branch=main)](https://travis-ci.com/ovro-lwa/distributed-pipeline)
[![codecov](https://codecov.io/gh/ovro-lwa/distributed-pipeline/branch/main/graph/badge.svg)](https://codecov.io/gh/ovro-lwa/distributed-pipeline)
[![Documentation Status](https://readthedocs.org/projects/distributed-pipeline/badge/?version=latest)](https://distributed-pipeline.readthedocs.io/en/latest/?badge=latest)
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

```
pipenv sync --dev
```
This should install the dependencies of the project (with versions etc as specified in `Pipfile.lock`). Then run the pipenv-managed virtualenv:
```pipenv shell```
or prefix any command with:
```pipenv run```


To run the tests, do
```
pytest
```
which should run the tests and output a report.


It is recommended that pycharm be used for development. I have not settled on a
style linter or a documentation format yet...You can sync your local codebase with astm by setting the `$ASTM_USER_NAME` variable and then run `./sync_with_astm.sh` to sync.

Adding a function to orca also requires integrating it with celery. This [example commit](https://github.com/ovro-lwa/distributed-pipeline/commit/e1e577437bef3c19162bdab1cd3973bee2128c04) shows the way to add and integrate a new function.

## Run with celery
TBA, but meanwhile see scripts in `proj` directory. celery admin notes are in `celery.md`.

## Code Structure
`orca` is where the wrappers and functions that do single units of work sit.

`proj` contains code that executes the pipeline with celery.

## Install new packages/dependencies
`pipenv.lock` is the definitive source of all the dependencies (it records the version number, the buid, etc).
When you call `pipenv sync`, it installs all the packages recorded in `pipenv.lock`. To add a new package as
dependency to the project, instead of calling `pip install`, do `pipenv install --keep-oudated` with the package
so that it installs the package, update the minimal set of packages required, and then write the current state
of the packages into `pipenv.lock`. Add a `--dev` flag to `instaill` if you only want the package in the `dev`
environment (say, if you're only gonna use this for your notebooks/offline analyses instead of your pipeline)

## Updating package
`pipenv install` is still the command to call. Say I want to upgrade numpy to 1.19.1, I'd do
```
pipenv install --keep-outdated 'numpy==1.19.1'
```
This would update both `Pipfile.lock` and `Pipfile`. I then usually do the following so that I don't spec these
packages in the `Pipfile` (so that the versions don't get accidentally locked)
```
git checkout -- Pipfile
```
Note that the new environment is still stored in `Pipfile.lock`.

`pipenv update` would try to update the entire environment, no matter what you call it with. I don't use it. It is
useful to see what's oudated with
```
pipenv update --dry-run
```
