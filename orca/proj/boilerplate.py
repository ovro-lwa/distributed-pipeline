from orca.flagging import merge_flags, flag_bad_chans
from orca.proj.celery import app
from orca.transform import dada2ms, change_phase_centre, peeling


@app.task
def run_dada2ms(dada_file, out_ms, gaintable=None):
    dada2ms.run_dada2ms(dada_file, out_ms, gaintable)


@app.task
def run_chgcentre(ms_file, direction):
    change_phase_centre.change_phase_center(ms_file, direction)


@app.task
def peel(ms_file, sources):
    # TODO make this idempotent somehow
    peeling.peel_with_ttcal(ms_file, sources)


@app.task
def apply_a_priori_flags(ms_file, flag_npy_path, create_corrected_data_column=False):
    merge_flags.write_to_flag_column(ms_file, flag_npy_path,
                                     create_corrected_data_column=create_corrected_data_column)


@app.task
def apply_ant_flag(ms_file, ants):
    from casacore.tables import table, taql
    t = table(ms_file)
    taql(f"update $t set FLAG=True where any(ANTENNA1==$ants || ANTENNA2==$ants)")


@app.task
def flag_chans(ms, spw):
    flag_bad_chans.flag_bad_chans(ms, spw, apply_flag=True)
    return ms