import pytest


# Test import so that celery workers wouldn't fail silently on shutdown.
def test_import():
    import orca.proj.boilerplate
