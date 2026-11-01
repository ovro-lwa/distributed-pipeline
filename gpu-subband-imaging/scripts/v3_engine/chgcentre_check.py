#!/usr/bin/env python
"""gsi v3 step 4: the in-memory LST rotation against chgcentre, on real averaged MSes (CPU).

v2 (``snapshot_phase_centre: lst``) copies each peeled MS and runs ``chgcentre <ms> <ra> <dec>``
before imaging.  The engine instead rotates in memory (v3/rotation.py, ttcalx_imager.rotate):

  exact  uvw' = A[a1] - A[a2] with A = rotate.measures_uvw(ant_pos, TIME[0], ra, dec)
         (per-antenna J2000 uvw from casacore measures, as chgcentre computes them);
  R      uvw' = R @ uvw, R = rotate.rotation_from_uvw(uvw, A[a1] - A[a2]) (the hour record's R);

and vis' = vis * exp(2 pi i (w - w') / lambda) in complex128, stored as complex64.  For every
MS this prints, against a chgcentre'd copy: max |uvw' - uvw_chgcentre| [m] for both forms,
max |vis' - vis_chgcentre| / rms(vis) for I and V (both forms), whether the flags and the
other columns are untouched, and the new PHASE_DIR against rotate.parse_phase_centre.

Imager venv python, PYTHONPATH=<imager src>:<pipeline src>:

    chgcentre_check.py --phase-centre "05h30m00s 37d12m57.057s" --work /tmp/x MS... [--json out]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

C = 299792458.0


def check(ms: str, centre: str, work: Path, chgcentre: str) -> dict:
    from casacore.tables import table
    from ttcalx_imager import rotate
    from ttcalx_imager.msio import read_ms

    ra, dec = rotate.parse_phase_centre(centre)
    cp = work / (Path(ms).name + ".chg")
    shutil.rmtree(cp, ignore_errors=True)
    shutil.copytree(ms, cp)
    r = subprocess.run([chgcentre, str(cp)] + centre.split(), capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stdout + r.stderr)
    with table(ms, ack=False) as t:
        a1, a2 = t.getcol("ANTENNA1"), t.getcol("ANTENNA2")
        uvw0, data0, flag0 = t.getcol("UVW"), t.getcol("DATA"), t.getcol("FLAG")
        tcol = t.getcol("TIME")
    with table(str(cp), ack=False) as t:
        uvwc, datac, flagc = t.getcol("UVW"), t.getcol("DATA"), t.getcol("FLAG")
        same_other = {c: bool(np.array_equal(t.getcol(c), table(ms, ack=False).getcol(c)))
                      for c in ("TIME", "WEIGHT", "SIGMA", "ANTENNA1", "ANTENNA2", "FLAG_ROW")}
    with table(str(cp) + "/FIELD", ack=False) as f:
        pd_c = f.getcol("PHASE_DIR")[0, 0]
    with table(ms + "/SPECTRAL_WINDOW", ack=False) as s:
        freqs = s.getcol("CHAN_FREQ")[0]
    with table(ms + "/ANTENNA", ack=False) as a:
        pos = a.getcol("POSITION")
    keep = a1 != a2
    A = rotate.measures_uvw(pos, float(tcol[0]), ra, dec)
    exact = A[a1] - A[a2]
    m = read_ms(ms)
    R = rotate.rotation_from_uvw(m.uvw, exact[keep])
    R2 = rotate.snapshot_rotation(m.uvw, m.a1, m.a2, m.ant_pos, m.time, ra, dec)
    viaR = uvw0 @ R.T
    out = dict(ms=ms, nrow=int(len(a1)), R_equals_snapshot_rotation=bool(np.array_equal(R, R2)),
               phase_dir_chg=[float(x) for x in pd_c], phase_dir_parse=[ra, dec],
               phase_dir_dmax=float(np.max(np.abs(pd_c - np.array([ra, dec])))),
               flags_untouched=bool(np.array_equal(flag0, flagc)), other_columns_untouched=same_other,
               autos_uvw_chg_max=float(np.abs(uvwc[~keep]).max()) if (~keep).any() else 0.0)
    lam = freqs / C
    dI_rms = float(np.sqrt(np.mean(np.abs(0.5 * (datac[keep][..., 0] + datac[keep][..., 3])) ** 2)))
    for name, un in (("exact", exact), ("R", viaR)):
        d = np.abs(un[keep] - uvwc[keep])
        ph = np.exp(2j * np.pi * (uvw0[:, 2:3] - un[:, 2:3]) * lam[None])          # (nrow, nchan)
        dm = (data0.astype(np.complex128) * ph[:, :, None]).astype(np.complex64)
        diff = np.abs(dm[keep] - datac[keep])
        I_m = 0.5 * (dm[keep][..., 0] + dm[keep][..., 3]); I_c = 0.5 * (datac[keep][..., 0] + datac[keep][..., 3])
        V_m = 0.5j * (dm[keep][..., 2] - dm[keep][..., 1]); V_c = 0.5j * (datac[keep][..., 2] - datac[keep][..., 1])
        out[name] = dict(uvw_max_m=float(d.max()), uvw_rms_m=float(np.sqrt((d ** 2).mean())),
                         w_max_m=float(d[:, 2].max()),
                         data_max_abs=float(diff.max()), data_rel_max=float(diff.max() / dI_rms),
                         data_rel_rms=float(np.sqrt((diff ** 2).mean()) / dI_rms),
                         data_bitwise_frac=float((dm[keep] == datac[keep]).mean()),
                         I_rel_max=float(np.abs(I_m - I_c).max() / dI_rms),
                         V_rel_max=float(np.abs(V_m - V_c).max() / dI_rms))
    shutil.rmtree(cp, ignore_errors=True)
    return out


def phase_dirs(ms: str, work: Path, chgcentre: str) -> list:
    """PHASE_DIR chgcentre writes for every LST-hour centre against parse_phase_centre(_chgcentre)."""
    from casacore.tables import table
    from ttcalx_imager import rotate
    cp = work / (Path(ms).name + ".pd")
    out = []
    for h in range(24):
        pc = f"{h:02d}h30m00s 37d12m57.057s"
        shutil.rmtree(cp, ignore_errors=True)
        shutil.copytree(ms, cp)
        r = subprocess.run([chgcentre, str(cp)] + pc.split(), capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stdout + r.stderr)
        with table(str(cp) + "/FIELD", ack=False) as f:
            pd = [float(x) for x in f.getcol("PHASE_DIR")[0, 0]]
        mine = rotate.parse_phase_centre_chgcentre(pc)
        astro = rotate.parse_phase_centre(pc)
        out.append(dict(pc=pc, chgcentre=pd, mine_equal=list(mine) == pd, astropy_equal=list(astro) == pd))
    shutil.rmtree(cp, ignore_errors=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("ms", nargs="+")
    ap.add_argument("--phase-centre", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--chgcentre", default="/opt/bin/chgcentre")
    ap.add_argument("--json", default="")
    ap.add_argument("--phase-dirs", action="store_true", help="only the 24 LST-hour PHASE_DIRs (first MS)")
    a = ap.parse_args(argv)
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    if a.phase_dirs:
        r = phase_dirs(a.ms[0], work, a.chgcentre)
        for x in r:
            print(json.dumps(x))
        print(f"mine equal {sum(x['mine_equal'] for x in r)}/24, astropy equal {sum(x['astropy_equal'] for x in r)}/24")
        return 0
    res = []
    for ms in a.ms:
        r = check(ms, a.phase_centre, work, a.chgcentre)
        res.append(r)
        print(json.dumps(r), flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
