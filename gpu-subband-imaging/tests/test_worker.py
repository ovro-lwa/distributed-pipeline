import pytest

from gpu_subband_imaging.worker import _require_batch_outputs


def test_batch_with_no_outputs_fails():
    with pytest.raises(RuntimeError, match="produced no outputs"):
        _require_batch_outputs(27, 4, n_input=45, n_ok=0)


def test_batch_with_any_output_can_continue():
    _require_batch_outputs(27, 4, n_input=45, n_ok=1)
