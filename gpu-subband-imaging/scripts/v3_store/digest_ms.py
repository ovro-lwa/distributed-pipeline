"""venv: digests of msio.load_samples for every MS of the bands (run with old or new imager src).
usage: digest_ms.py <out.json> <lst_hour> band..."""
import json, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from equiv_common import digest
import ttcalx_imager
from ttcalx_imager.msio import load_samples
try:
    from ttcalx_imager.rotate import parse_phase_centre
except ImportError:      # imager before the v3 change: hour.parse_phase_centre (hour.py needs cufinufft) -- same code
    def parse_phase_centre(pc):
        from astropy.coordinates import Angle
        a, b = pc.split()
        ra = Angle(a, unit="hourangle") if any(c in a for c in "hm:") else Angle(float(a), unit="deg")
        return ra.rad, Angle(b, unit="deg").rad
from ttcalx_imager import bands
print("imager:", ttcalx_imager.__file__, flush=True)
out, lh = sys.argv[1], int(sys.argv[2])
ra, dec = parse_phase_centre(bands.hour_phase_centre(lh))
res = {}
for band in sys.argv[3:]:
    for m in sorted(Path(os.environ.get("MS_ROOT", "/tmp/nk_v3_store/ms")) / f"{band}MHz".glob("*.ms")):
        s = load_samples(str(m), ra, dec)
        res[s["stem"]] = digest(s)
json.dump(dict(imager=ttcalx_imager.__file__, digests=res), open(out, "w"))
print("digests", len(res))
