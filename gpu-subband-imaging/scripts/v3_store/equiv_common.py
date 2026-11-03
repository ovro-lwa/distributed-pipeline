import hashlib, struct
import numpy as np

KEYS = ("path", "stem", "time", "tstart", "freqs", "chan_width", "pd", "R", "pd_new", "nrow", "nchan",
        "flagfrac", "zero_frac", "telescope", "observer", "field_name", "u", "v", "w", "nw", "I", "V")


def _bits(x):
    """Exact representation of a value: arrays by dtype/shape/bytes, floats by their IEEE bits."""
    if isinstance(x, np.ndarray):
        return ("nd", x.dtype.str, x.shape, hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest())
    if isinstance(x, (tuple, list)):
        return ("seq", type(x).__name__, tuple(_bits(v) for v in x))
    if isinstance(x, float):
        return ("float", struct.pack("<d", x).hex())
    if isinstance(x, np.floating):
        return ("npfloat", x.dtype.str, x.tobytes().hex())
    return (type(x).__name__, repr(x))


def digest(s, keys=KEYS, skip=()):
    return {k: _bits(s[k]) for k in keys if k not in skip}


def diff(a, b, keys=KEYS, skip=()):
    """Keys whose values are not bit-identical (arrays compared by bytes)."""
    bad = []
    for k in keys:
        if k in skip:
            continue
        x, y = a[k], b[k]
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            x, y = np.asarray(x), np.asarray(y)
            if x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes():
                bad.append(k)
        elif _bits(x) != _bits(y):
            bad.append(k)
    return bad
