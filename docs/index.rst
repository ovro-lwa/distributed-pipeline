orca
====

A distributed data processing pipeline for the OVRO Long Wavelength Array.

**orca** provides tools for:

* RFI flagging and data quality assessment
* Calibration and bandpass correction  
* Frequency and time averaging
* Dynamic spectrum generation
* Imaging and source finding

The pipeline uses **Celery** for distributed task execution across a compute cluster,
enabling parallel processing of large radio astronomy datasets.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   usage_guide

.. toctree::
   :maxdepth: 2
   :caption: Subband Pipeline

   subband-pipeline
   worker-management
   celery_deployment

.. toctree::
   :maxdepth: 3
   :caption: API Reference

   autoapi/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
