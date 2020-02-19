# orca
## Set up development environment
Run in this directory (with pipenv installed)
```
pipenv update --dev
```
This should install the dependencies of the project. Then run in the pipenv-managed virtualenv
(which can be invoked with `pipenv shell` or prefix the command with `pipenv run`)

Then run
```
pipenv install -e .
```
, which will install the package in development mode, so that libraries can be called and
binaries be executed. This can be re-run after code changes.

To run the tests, do
```
pytest
```
which should run the tests and output a report.

It is recommended that pycharm be used for development. I have not settled on a
style linter or a documentation format yet...

## Run with celery
TBA, but meanwhile see scripts in `proj` directory. celery admin notes are in `celery.md`.

## Code Structure
`orca` is where the wrappers and functions that do single units of work sit.

`proj` contains code that executes the pipeline with celery.
