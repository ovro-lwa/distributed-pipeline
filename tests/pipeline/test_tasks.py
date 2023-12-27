import pytest

from ..common import TEST_MS


# Test import so that celery workers wouldn't fail silently on shutdown.
def test_import():
    import orca.pipeline.tasks


def test_casa_sanity():
    from casatasks import listobs
    listobs(TEST_MS)