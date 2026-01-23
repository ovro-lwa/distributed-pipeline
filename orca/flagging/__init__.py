"""Flagging operations for OVRO-LWA measurement sets.

This subpackage provides functions for identifying and applying flags to
radio interferometry data, including:

- Antenna-based flagging from autocorrelation statistics
- Baseline-based flagging from correlations
- Channel-based flagging from amplitude statistics
- Flag file I/O operations

All flagging functions operate on CASA-format measurement sets using
casacore.tables for data access.
"""
