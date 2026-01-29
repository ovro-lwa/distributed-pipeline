"""Data transformation operations for OVRO-LWA visibility data.

This subpackage provides functions for transforming measurement set data,
including:

- Calibration application (bandpass, gain scaling)
- Time and frequency averaging
- Imaging with WSClean
- Dynamic spectrum generation
- Source peeling
- Quality assurance metrics

Most functions operate on CASA-format measurement sets and return the
path to the modified or output file.
"""
