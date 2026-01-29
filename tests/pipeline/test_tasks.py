import pytest

from ..common import TEST_MS


# Test import so that celery workers wouldn't fail silently on shutdown.
@pytest.mark.skip(reason="orca.pipeline module does not exist - pipeline/ is not a subpackage of orca")
def test_import():
    import orca.pipeline.tasks


@pytest.mark.skip(reason="CASA does not recognize OVRO-LWA telescope in the test MS")
def test_casa_sanity():
    from casatasks import listobs
    listobs(TEST_MS)