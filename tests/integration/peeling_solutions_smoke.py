"""Manual bounded server check. No Celery import, dispatch, or environment changes.

Run: python peeling_solutions_smoke.py REPO_ROOT TEST_MS SCRATCH_DIR
Uses fresh 8-antenna copies of the repository fixture, never production MSes.
"""
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import numpy as np
from casacore.tables import table

repo, fixture, scratch = map(Path, sys.argv[1:4])
scratch.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location('ttcal_export', repo / 'orca/wrapper/ttcal.py')
ttcal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ttcal)

sky_model = [
    dict(name='below', ra='00h00m00s', dec='-89d00m00s', I=100., freq=40e6, index=[0.]),
    dict(name='above', ra='00h00m00s', dec='+89d00m00s', I=100., freq=40e6, index=[0.]),
]
# Synthetic near-field source, independent of the private operational catalog.
rfi_model = [dict(name='synthetic-rfi', sys='WGS84', long=-118.38,
                  lat=37.30, el=1200.,
                  **{'rfi-frequencies': [1e6, 100e6], 'rfi-I': [5., 5.]})]

for env_name, model in [('julia060', sky_model), ('ttcal_dev', rfi_model)]:
    if len(sys.argv) > 4 and env_name not in sys.argv[4:]:
        continue
    root = scratch / env_name
    root.mkdir()
    src = root / 'sources.json'
    src.write_text(json.dumps(model))
    baseline = root / 'baseline.ms'
    exported = root / 'exported.ms'
    with table(str(fixture), ack=False) as t:
        with t.query('ANTENNA1 < 8 && ANTENNA2 < 8') as subset:
            subset.copy(str(baseline), deep=True, valuecopy=True).close()
    with table(str(baseline / 'ANTENNA'), readonly=False, ack=False) as ants:
        ants.removerows(list(range(8, ants.nrows())))
    with table(str(baseline), readonly=False, ack=False) as t:
        assert t.nrows() == 36
        # Exercise the Phase 1 CORRECTED_DATA branch as well as DATA fallback.
        if env_name == 'julia060':
            desc = t.getcoldesc('DATA')
            desc['name'] = 'CORRECTED_DATA'
            t.addcols(desc)
            t.putcol('CORRECTED_DATA', t.getcol('DATA'))
        before_flags = t.getcol('FLAG')
        before_data = t.getcol('DATA')
    shutil.copytree(baseline, exported)
    command = [f'/opt/devel/pipeline/envs/{env_name}/bin/julia',
               f'/opt/devel/pipeline/envs/{env_name}/bin/ttcal.jl',
               'zest', str(baseline), str(src), '--beam', 'constant',
               '--minuvw', '5', '--maxiter', '5', '--tolerance', '1e-4']
    env = os.environ.copy()
    if env_name == 'julia060':
        env.update(LD_LIBRARY_PATH='/opt/astro/mwe/usr/lib64:/opt/astro/lib/',
                   AIPSPATH='/opt/astro/casa-data dummy dummy')
    else:
        env['OMP_NUM_THREADS'] = '8'
        command = ['/bin/bash', '-c', 'source ~/.bashrc && conda activate ttcal_dev && exec '
                   + ' '.join(map(shlex.quote, command))]
    subprocess.run(command, env=env, check=True)
    output = root / 'solutions.npz'
    ttcal.zest_with_ttcal(str(exported), str(src), beam='constant', minuvw=5,
                         maxiter=5, solutions_path=str(output), julia_env=env_name)
    column = 'CORRECTED_DATA' if env_name == 'julia060' else 'DATA'
    with table(str(baseline), ack=False) as old, table(str(exported), ack=False) as new:
        np.testing.assert_allclose(new.getcol(column), old.getcol(column), rtol=0, atol=0)
        np.testing.assert_array_equal(new.getcol('FLAG'), before_flags)
        if column == 'CORRECTED_DATA':
            np.testing.assert_array_equal(new.getcol('DATA'), before_data)
    with np.load(output, allow_pickle=False) as z:
        assert z['gains'].shape == (1, 4, 8, 12, 1)
        assert z['source_indices'].tolist() == ([1] if env_name == 'julia060' else [0])
        assert np.isfinite(z['gains']).all()
        print('PASS', env_name, 'identical visibilities; shape', z['gains'].shape,
              'compressed bytes', output.stat().st_size, flush=True)
