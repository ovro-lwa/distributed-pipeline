from gpu_subband_imaging.stages.stage_ms import stage


def test_stage_copies_raw_measurement_set_to_scratch(tmp_path):
    source = tmp_path / "source" / "20260419_050000_73MHz.ms"
    source.mkdir(parents=True)
    (source / "table.dat").write_text("original")
    work = tmp_path / "work"

    staged = stage(str(source), work)

    assert staged == work / source.name
    assert (staged / "table.dat").read_text() == "original"
    (staged / "table.dat").write_text("changed in scratch")
    assert (source / "table.dat").read_text() == "original"
