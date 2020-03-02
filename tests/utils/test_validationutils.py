import pytest

from orca.utils import validationutils


def test_empty_tuple():
    with pytest.raises(ValueError):
        validationutils.check_collection_not_empty(())


def test_empty_dict():
    with pytest.raises(ValueError):
        validationutils.check_collection_not_empty({})


def test_empty_list():
    with pytest.raises(ValueError):
        validationutils.check_collection_not_empty([])


def test_nonempty_list():
    validationutils.check_collection_not_empty([2])
