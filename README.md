# orca
[![Build Status](https://travis-ci.com/ovro-lwa/distributed-pipeline.svg?branch=main)](https://travis-ci.com/ovro-lwa/distributed-pipeline)
[![codecov](https://codecov.io/gh/ovro-lwa/distributed-pipeline/branch/main/graph/badge.svg)](https://codecov.io/gh/ovro-lwa/distributed-pipeline)
[![Documentation Status](https://readthedocs.org/projects/distributed-pipeline/badge/?version=latest)](https://distributed-pipeline.readthedocs.io/en/latest/?badge=latest)

## Set up development environment on calim
If you have not done so, create a barebone python3.8 environment with conda (on calim there's a py38_orca environment that you can just use).
You will need conda-forge in your channels to install pipenv. If it's not there, you can use the `-c conda-forge` flag.
```
conda create --name py38_orca python=3.8 pipenv
```

Activate with
```
conda activate py38_orca
```
Then checkout this repo and install
```
pip install -r requirements.txt
```

To run the tests, do
```
pytest
```
which should run the tests and output a report.

Adding a function to orca also requires integrating it with celery. This [example commit](https://github.com/ovro-lwa/distributed-pipeline/commit/e1e577437bef3c19162bdab1cd3973bee2128c04) shows the way to add and integrate a new function. A good way to develop code for celery is to create a function with a unit test. These can be implemented in a single module. An function can be made into a task with the celery application decorator `@app.task` (`app` is imported from the `celery.py` module in this repo).

## Run with celery
After code is installed, you can run the celery application, which is a persisent process that manages the queue:

```
celery -A orca.celery worker -Q default
```

Now you can submit tasks to the application from another session (e.g., IPython, notebook, etc). A common way to submit a task is to use the `delay` property, so for your decorated function `do_something(a, b)`, you can run it as `result = do_something.delay(a, b)`.  The object `result` will refer to the task running asynchronously. You can use properties on `result` to see the status and return value of the function.

Celery admin notes are in `celery.md`. The submission session will show some logging, but the celery application process will show more.

## Code Structure
`orca` is where the wrappers and functions that do single units of work sit.

`pipeline` has useful syntax for composing tasks, setting up arguments for many jobs, and processing results.

## Installing new packages/dependencies or updating packages

`pip install` is still the command to call. Say I want to upgrade numpy to 1.19.1, I'd do
```
pip install 'numpy==1.19.1'
```
