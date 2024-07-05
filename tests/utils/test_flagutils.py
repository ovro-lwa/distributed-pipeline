import pytest

from orca.utils.flagutils import FLAG_TABLE, get_bad_ants
from datetime import date
from os import path

@pytest.mark.skipif(not path.isdir(FLAG_TABLE), reason="need acual data.")
def test_get_bad_ants():
    assert get_bad_ants(date(2024, 5, 19)) == [3, 12, 14, 17, 28, 33, 41, 44, 79, 80, 87, 117, 126, 137, 150, 154, 178, 193, 201, 208, 211, 215, 218, 224, 230, 231, 236, 242, 246, 261, 289, 294, 301, 307, 309, 331]
