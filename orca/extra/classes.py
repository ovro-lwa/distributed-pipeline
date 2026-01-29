"""Classification labels for detected sources.

Defines enumeration classes for categorizing detected sources
in transient and source finding pipelines.
"""
from enum import IntEnum


class Classes(IntEnum):
    """Source classification labels for transient detection.

    Attributes:
        NA: Not assigned / unknown.
        CANDIDATE: Potential transient candidate.
        AIRPLANE: Aircraft reflection.
        METEOR: Meteor trail.
        SCINT: Scintillation artifact.
        OTHER: Other classification.
        REFRACTION: Ionospheric refraction artifact.
        STATIONARY: Stationary source (not transient).
        SUN: Solar emission.
        ATEAM: Known A-team source (CasA, CygA, etc.).
    """
    NA = 0
    CANDIDATE = 1
    AIRPLANE = 2
    METEOR = 3
    SCINT = 6
    OTHER = 7
    REFRACTION = 8
    STATIONARY = 9
    SUN = 10
    ATEAM = 11
