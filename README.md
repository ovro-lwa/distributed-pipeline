# ORCA — Tools for distributed OVRO-LWA data processing

[![Tests](https://github.com/ovro-lwa/distributed-pipeline/actions/workflows/tests.yml/badge.svg)](https://github.com/ovro-lwa/distributed-pipeline/actions/workflows/tests.yml)
[![Docs](https://github.com/ovro-lwa/distributed-pipeline/actions/workflows/docs.yml/badge.svg)](https://github.com/ovro-lwa/distributed-pipeline/actions/workflows/docs.yml)

## Table of Contents
- [Installation and Environment Setup](#installation-and-environment-setup)
- [Additional Software Requirements](#additional-software-requirements)
- [Configuration Setup](#configuration-setup)
- [Run with celery](#run-with-celery)
- [Code Structure](#code-structure)
- [Developer & Testing Guide](#developer--testing-guide)

## Installation and Environment Setup

This repository serves as the central place for running and developing the OVRO-LWA data reduction and calibration pipelines.  
A pre-configured environment is already available on Calim — see the [Developer & Testing Guide](#developer--testing-guide) for instructions and usage examples.  
The following are instructions for setting up a new environment from scratch.


First, clone the repository:

```
git clone https://github.com/ovro-lwa/distributed-pipeline.git
cd distributed-pipeline
```

Then create the environment using the provided `environment.yaml`:

```
conda env create -f environment.yaml
```

Activate the environment:

```
conda activate orca-env
```

All required dependencies — including both conda and pip packages — are installed automatically by `environment.yaml`, so no additional installation commands are needed.

## Additional Software Requirements

In addition to the Python packages listed in `requirements.txt`, several external tools are required to run certain ORCA functions:

- **WSClean** — used for imaging. Also includes tools like `chgcentre` for changing the phase center of a measurement set. See:
  - [WSClean GitLab repository](https://gitlab.com/aroffringa/wsclean)
  - [WSClean documentation](https://wsclean.readthedocs.io/en/latest/)

- **AOFlagger** — used for RFI flagging. See:
  - [AOFlagger GitLab repository](https://gitlab.com/aroffringa/wsclean)
  - [AOFlagger documentation](https://aoflagger.readthedocs.io/en/latest/)

- **TTCal.jl** — used for peeling bright sources: [http://mweastwood.info/TTCal.jl/](http://mweastwood.info/TTCal.jl/)

- **mnc_python** — OVRO-LWA Monitor and Control Python tools, used in functions like automatic bad antenna detection:  
  [https://github.com/ovro-lwa/mnc_python](https://github.com/ovro-lwa/mnc_python)


## Configuration Setup

Copy the default configuration file to your home directory:

```bash
cp orca/default-orca-conf.yml ~/orca-conf.yml
```

If you plan to use Celery, edit the `queue:` section in `~/orca-conf.yml` and update:

- `broker_uri` with your RabbitMQ URI
- `result_backend_uri` with your Redis backend address

If you are not using Celery, you can leave the `queue:` section unchanged.  
It will not affect functionality unless Celery-based task execution is used.

The configuration file is still required for settings related to telescope layout and executable paths.

## Run with Celery
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






