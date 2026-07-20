from gpu_subband_imaging.worklist import _chunk


def test_chunk_splits_evenly_and_remainder():
    assert _chunk(list(range(10)), 4) == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]


def test_chunk_empty():
    assert _chunk([], 50) == []
