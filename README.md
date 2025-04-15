# ORCA — Tools for distributed OVRO-LWA data processing

## Table of Contents
- [Installation and Environment Setup](#installation-and-environment-setup)
- [Run with celery](#run-with-celery)
- [Code Structure](#code-structure)
- [Developer & Testing Guide](#developer--testing-guide)

## Installation and Environment Setup

This repository serves as the central place for running and developing the OVRO-LWA data reduction and calibration pipelines.  
A pre-configured environment is already available on Calim — see the [Developer & Testing Guide](#developer--testing-guide) for instructions and usage examples.  
The following are instructions for setting up a new environment from scratch.


If you have not done so, create a barebone python3.8 environment with conda.
```
conda create --name py38_orca python=3.8 pipenv
```

Activate with

```
conda activate py38_orca
```

Then checkout this repo:

```
git clone https://github.com/ovro-lwa/distributed-pipeline.git
cd distributed-pipeline
```

Then install dependencies:

```
pip install -r requirements.txt
pip install .
```

Copy `orca/default-orca-conf.yml` to your home directory and modify it with the correct rabbitMQ URI if you plan to run with celery. Otherwise you can run things as plain ole functions.

## Run with celery
Adding a function to orca also requires integrating it with celery. This [example commit](https://github.com/ovro-lwa/distributed-pipeline/commit/e1e577437bef3c19162bdab1cd3973bee2128c04) shows the way to add and integrate a new function. A good way to develop code for celery is to create a function with a unit test An function can be made into a task with the celery application decorator `@app.task` (`app` is imported from the `celery.py` module in this repo). You can call the decorated function like a regular function, test it locally, etc.

This is for integration testing only. Make sure you read https://docs.celeryq.dev/en/stable/getting-started/introduction.html before you start.

You can start a celery worker with
```
celery -A orca.celery worker -Q default --name=<give-it-a-name> --loglevel=INFO
```

The queue and backend are configured in `celery.py` under `orca`. Make sure you use a different rabbitMQ vhost for testing.

Now you can submit tasks to the application from another session (e.g., IPython, notebook, etc). A common way to submit a task is to use the `delay` member function, so for your decorated function `do_something(a, b)`, you can run it as `result = do_something.delay(a, b)`.  The object `result` will refer to the task running asynchronously. You can use properties on `result` to see the status and return value of the function.

Celery admin notes are in `celery.md`. The submission session will show some logging, but the celery application process will show more.

## Code Structure
`orca` is where the wrappers and functions that do single units of work sit.

`pipeline` is where the pipelines live and serve as useful examples for how to use celery.

## Developer & Testing Guide

For usage examples and how to test the pipeline without Celery, please refer to the [Usage Guide](usage_guide.md)





