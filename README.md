# astm-transient-stuff
## Set up development environment
Run in this directory (with pipenv properly installed)
```
pipenv update --dev
```
This should install the dependencies of the project. Then run in the pipenv-managed virtualenv
(which can be invoked with `pipenv shell` or prefix the command with `pipenv run`)
```
python ./setup.py test
```

to run the tests.

Optionally,
```
pipenv install -e .
```
will install the package in development mode, so that libraries can be called and
binaries be executed.

It is recommended that pycharm be used for development. I have not settled on a
style linter or a documentation format yet...

## Run with celery
I put some of the celery management commands in celery.md. But to run things with
celery. Do the following.

## Code Structure
`orca` is where the wrappers and functions that do single units of work sit.

`proj` contains code that executes the pipeline with celery.

`notebook` contains ad-hoc analyses and experimental things.