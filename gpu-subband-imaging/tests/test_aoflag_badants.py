"""v3.aoflag (AOFlagger after peeling on the band-hour) and the nightly's pinned bad-antenna list (no GPU,
no CASA, no AOFlagger: a stand-in module replaces `aoflagger`)."""
import importlib.util
import sys
import types
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import yaml

from gpu_subband_imaging.config import ConfigError, V3Params
from gpu_subband_imaging.v3 import aoflag_hour, slots
from test_v3_slots import make_snap


def _nightly():
    p = Path(__file__).resolve().parents[1] / "scripts" / "run_nightly_lst.py"
    spec = importlib.util.spec_from_file_location("run_nightly_lst_t", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# -- AOFlagger on the band-hour ---------------------------------------------------------------------------------
class _Mask:
    def __init__(self, w, h, v):
        self.b = np.full((h, w), v, dtype=bool)

    def set_buffer(self, b):
        self.b = np.array(b, dtype=bool)

    def get_buffer(self):
        return self.b


class _ImageSet:
    def __init__(self, w, h, n):
        self.bufs = [None] * n

    def set_image_buffer(self, i, b):
        self.bufs[i] = np.array(b)


class _Strategy:
    """Flags (channel, time) where |XX| (real image 0 + imaginary image 1) exceeds 3, plus the input mask."""
    def run(self, iset, mask):
        out = _Mask(1, 1, False)
        out.set_buffer(np.hypot(iset.bufs[0], iset.bufs[1]) > 3)
        return out


class _AOFlagger:
    def load_strategy_file(self, path):
        return _Strategy()

    def make_image_set(self, w, h, n):
        return _ImageSet(w, h, n)

    def make_flag_mask(self, w, h, v):
        return _Mask(w, h, v)


@pytest.fixture
def fake_aoflagger(monkeypatch):
    monkeypatch.setitem(sys.modules, "aoflagger", types.SimpleNamespace(AOFlagger=_AOFlagger))


def _hour(tmp_path, n=4, nant=6, nchan=5):
    paths = []
    for t in range(n):
        s = make_snap(nant=nant, nchan=nchan, seed=t, stem=f"2026111{t}_000000_55MHz")
        s.data[np.isnan(s.data.real) | np.isinf(s.data.imag)] = 0
        p = tmp_path / f"{t}.slot"
        slots.write_slot(p, s)
        paths.append(str(p))
    return paths


def test_aoflag_hour_ors_flags_into_all_correlations(tmp_path, fake_aoflagger):
    paths = _hour(tmp_path)
    before = [slots.view_slot(p, writable=False) for p in paths]
    f0 = [np.array(s.flag) for s in before]
    d0 = [np.array(s.data) for s in before]
    hot = (0 + 1j) * 10                                   # one RFI sample: snapshot 2, the first cross row, chan 3
    cross = int(np.flatnonzero(before[0].antenna1 != before[0].antenna2)[0])
    s2 = slots.view_slot(paths[2], writable=True)
    s2.data[cross, 3, 0] = hot
    d0[2][cross, 3, 0] = hot
    del s2
    stats = aoflag_hour.flag_hour(paths, "strategy.lua", procs=1)
    for t, p in enumerate(paths):
        s = slots.view_slot(p, writable=False)
        a1, a2 = np.asarray(s.antenna1), np.asarray(s.antenna2)
        auto = a1 == a2
        assert np.array_equal(s.flag[auto], f0[t][auto])                     # autos untouched
        assert np.array_equal(np.asarray(s.data), d0[t])                     # data untouched
        assert (s.flag >= f0[t]).all()                                       # flags only added
        x = np.flatnonzero(~auto)
        merged = f0[t][x].any(axis=2)                                        # an initial flag in any correlation
        new = np.abs(d0[t][x, :, 0]) > 3
        assert np.array_equal(s.flag[x], np.repeat((merged | new)[:, :, None], 4, axis=2))
    assert slots.view_slot(paths[2], writable=False).flag[cross, 3].all()
    assert stats["n_snap"] == 4 and stats["flag_frac_after"] >= stats["flag_frac_before"]


def test_aoflag_hour_processes_give_the_same_flags(tmp_path, fake_aoflagger):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    pa, pb = _hour(a), _hour(b)
    aoflag_hour.flag_hour(pa, "s.lua", procs=1)
    for k, part in enumerate(np.array_split(aoflag_hour._rows(pb)[0], 3)):   # rows split as the processes do
        aoflag_hour._work((pb, "s.lua", part))
    for x, y in zip(pa, pb):
        assert np.array_equal(slots.view_slot(x).flag, slots.view_slot(y).flag)


def test_aoflag_hour_refuses_mixed_baselines(tmp_path):
    paths = _hour(tmp_path, n=2)
    slots.write_slot(tmp_path / "x.slot", make_snap(nant=7, nchan=5))
    with pytest.raises(ValueError, match="rows / channels differ"):
        aoflag_hour._rows(paths + [str(tmp_path / "x.slot")])


_MEM = dict(prep_inmem="on", peel_io="shm", snapshot_io="shm", hour_source="memory")


def test_v3_aoflag_config():
    assert V3Params().aoflag == "off"
    assert V3Params(aoflag=False).aoflag == "off"                                   # unquoted YAML off
    assert V3Params(aoflag=True, **_MEM).aoflag == "hour"                           # unquoted YAML on
    assert V3Params(aoflag="hour", **_MEM).write_kept_ms is False
    with pytest.raises(ConfigError):
        V3Params(aoflag="maybe")
    for bad in (0, True, "4"):
        with pytest.raises(ConfigError):
            V3Params(aoflag_procs=bad)


def test_v3_aoflag_needs_the_memory_hour_source():
    # pass 1 drops the raw MS, so kept MSes for a kept_ms / archive hour stage would silently be missing
    with pytest.raises(ConfigError, match="hour_source=memory"):
        V3Params(aoflag="hour", prep_inmem="on", peel_io="shm", snapshot_io="shm", hour_source="kept_ms")
    with pytest.raises(ConfigError, match="write_kept_ms"):
        V3Params(aoflag="hour", write_kept_ms=True, **_MEM)


# -- one bad-antenna list per night ---------------------------------------------------------------------------
def test_pin_badants_waits_for_the_night_record(monkeypatch):
    m = _nightly()
    from gpu_subband_imaging.stages import flag
    from astropy.time import Time
    rec = {"mjd": Time("2026-11-13T08:00:08").mjd}
    monkeypatch.setattr(flag, "badants_record", lambda mjd, py: (rec["mjd"], "3,7,11"))
    now = datetime(2026, 11, 14, 14, 0)
    monkeypatch.setattr(m, "utcnow", lambda: now)
    late = datetime(2026, 11, 14, 16, 30)
    assert m.pin_badants("2026-11-14", "py", late) is None                 # only yesterday's record so far
    rec["mjd"] = Time("2026-11-14T08:00:06").mjd
    b = m.pin_badants("2026-11-14", "py", late)
    assert b["own_night"] and b["csv"] == "3,7,11" and b["n"] == 3 and b["record_utc"].startswith("2026-11-14")


def test_pin_badants_after_the_deadline(monkeypatch):
    m = _nightly()
    from gpu_subband_imaging.stages import flag
    from astropy.time import Time
    monkeypatch.setattr(flag, "badants_record", lambda mjd, py: (Time("2026-11-13T08:00:08").mjd, "5"))
    monkeypatch.setattr(m, "utcnow", lambda: datetime(2026, 11, 14, 17, 0))
    b = m.pin_badants("2026-11-14", "py", datetime(2026, 11, 14, 16, 30))
    assert b is not None and not b["own_night"] and b["csv"] == "5"
    monkeypatch.setattr(flag, "badants_record", lambda mjd, py: (None, ""))      # lookup fails past the deadline:
    b = m.pin_badants("2026-11-14", "py", datetime(2026, 11, 14, 16, 30))          # not pinned, the night goes on
    assert b is not None and b["pinned"] is False and b["csv"] is None


def test_pin_badants_lookup_failure_before_the_deadline_waits(monkeypatch):
    m = _nightly()
    from gpu_subband_imaging.stages import flag
    monkeypatch.setattr(flag, "badants_record", lambda mjd, py: (None, ""))
    monkeypatch.setattr(m, "utcnow", lambda: datetime(2026, 11, 14, 14, 0))
    assert m.pin_badants("2026-11-14", "py", datetime(2026, 11, 14, 16, 30)) is None


def test_pin_badants_empty_own_record_is_valid(monkeypatch):
    m = _nightly()
    from gpu_subband_imaging.stages import flag
    from astropy.time import Time
    monkeypatch.setattr(flag, "badants_record", lambda mjd, py: (Time("2026-11-14T08:00:06").mjd, ""))
    monkeypatch.setattr(m, "utcnow", lambda: datetime(2026, 11, 14, 15, 40))
    b = m.pin_badants("2026-11-14", "py", datetime(2026, 11, 14, 16, 30))
    assert b["pinned"] and b["own_night"] and b["csv"] == "" and b["n"] == 0


def test_badants_record_survives_a_missing_python():
    from gpu_subband_imaging.stages import flag
    assert flag.badants_record(60000.0, "/nonexistent/python") == (None, "")


def test_write_config_pins_badants(tmp_path):
    m = _nightly()
    tpl = tmp_path / "tpl"
    tpl.mkdir()
    for n in ("cluster.yaml", "subbands.yaml"):
        (tpl / n).write_text("{}\n")
    (tpl / "pipeline.yaml").write_text(yaml.safe_dump({"cal": {"bp": "x"}, "description": "nightly"}))
    b = dict(csv="1,2", n=2, record_utc="2026-11-14T08:00:06", own_night=True)
    out = m.write_config(tpl, tmp_path / "out", "2026-11-14", [3, 4], "Run_x", "bp", b)
    d = yaml.safe_load((out / "pipeline.yaml").read_text())
    assert d["cal"]["badants"] == "1,2" and d["cal"]["bp"] == "bp"
    assert "2026-11-14T08:00:06" in d["description"] and d["description"].startswith("nightly")
    out2 = m.write_config(tpl, tmp_path / "out2", "2026-11-14", [3], "Run_x", "bp")
    assert "badants" not in yaml.safe_load((out2 / "pipeline.yaml").read_text())["cal"]
    out3 = m.write_config(tpl, tmp_path / "out3", "2026-11-14", [3], "Run_x", "bp", dict(pinned=False, csv=None))
    d3 = yaml.safe_load((out3 / "pipeline.yaml").read_text())
    assert "badants" not in d3["cal"] and "NOT pinned" in d3["description"]
    out4 = m.write_config(tpl, tmp_path / "out4", "2026-11-14", [3], "Run_x", "bp",
                          dict(pinned=True, csv="", n=0, record_utc="2026-11-14T08:00:06", own_night=True))
    assert yaml.safe_load((out4 / "pipeline.yaml").read_text())["cal"]["badants"] == ""


def test_fully_flagged_snapshot_is_empty_not_failed():
    from gpu_subband_imaging.v3.worker_v3 import _image_empty
    assert _image_empty({"status": "ok", "image": {"status": "empty", "files": []}})
    assert not _image_empty({"status": "ok", "image": {"status": "ok", "files": ["a", "b"]}})
    assert not _image_empty({"status": "ok"})                      # replay: no image requested
