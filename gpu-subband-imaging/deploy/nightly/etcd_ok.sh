#!/bin/bash
# exit 0 when the MNC etcd anthealth service answers with a non-empty bad-antenna list (2026-11-21: etcd down,
# calibration and the bad-antenna lookups failed); the real-time stream and the nightly wait for this
timeout 120 /opt/devel/pipeline/envs/development/bin/python -c "
import sys
from astropy.time import Time
from mnc import anthealth
c, bad = anthealth.get_badants(\"selfcorr\", time=Time.now().mjd)
sys.exit(0 if c is not None and len(bad) > 0 else 1)" > /dev/null 2>&1
