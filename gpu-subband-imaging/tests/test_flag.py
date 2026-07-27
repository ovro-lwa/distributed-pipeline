import subprocess

from gpu_subband_imaging.stages import flag


def test_badants_ignores_helper_stdout(monkeypatch):
    stdout = "Read antpos from default source\nGSI_BADANTS=12,34\nmore chatter\n"
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, stdout, ""),
    )

    assert flag.badants(60000.0, "/dev/python") == "12,34"


def test_badants_accepts_an_empty_result(monkeypatch):
    stdout = "Read antpos from default source\nGSI_BADANTS=\n"
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, stdout, ""),
    )

    assert flag.badants(60000.0, "/dev/python") == ""


def test_badants_does_not_return_unmarked_stdout(monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, "Read antpos from default source\n", ""
        ),
    )

    assert flag.badants(60000.0, "/dev/python") == ""
