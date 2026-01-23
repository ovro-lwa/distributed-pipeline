"""ORCA: OVRO-LWA Radio Camera Analysis Pipeline.

The orca package provides a distributed data processing pipeline for the
OVRO Long Wavelength Array (OVRO-LWA) radio telescope. It is built on top
of Celery for distributed task execution and provides tools for:

- Measurement set calibration and flagging
- Dynamic spectrum generation
- Imaging with WSClean
- Source peeling with TTCal
- Quality assurance and diagnostics

Subpackages
-----------
calibration
    Bandpass and delay calibration pipelines.
extra
    Additional utilities for source finding and catalog handling.
flagging
    Antenna, baseline, and channel flagging operations.
metadata
    Path management for measurement sets and data products.
resources
    Static data files and configuration maps.
tasks
    Celery task definitions for distributed processing.
transform
    Data transformation operations (calibration, imaging, averaging).
utils
    General utility functions and helpers.
wrapper
    Wrappers around external tools (WSClean, TTCal, AOFlagger).
"""
