from gpu_subband_imaging.stages.casarun import run_py


def test_run_py_isolates_casa_logging_from_shared_home():
    output = run_py(
        "import os\n"
        "from pathlib import Path\n"
        "home = Path(os.environ['HOME'])\n"
        "assert Path.cwd().resolve() == home.resolve()\n"
        "config = (home / '.casa' / 'config.py').read_text()\n"
        "assert 'logfile = \"/dev/null\"' in config\n"
        "assert 'telemetry_enabled = False' in config\n"
        "assert 'crashreporter_enabled = False' in config\n"
        "print('isolated')\n"
    )

    assert output.strip() == "isolated"
