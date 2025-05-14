from typing import List
from casacore.tables import table
import clickhouse_connect
import numpy as np
from orca.celery import app
from celery.utils.log import get_task_logger

from datetime import datetime
import os
import numpy.ma as ma
import casacore.tables as tables
import matplotlib.pyplot as plt
from casatasks import flagdata


logger = get_task_logger(__name__)

@app.task
def sanity_check(msl) -> List[str]:
    client = clickhouse_connect.get_client(host='10.41.0.85', username=os.getenv('CH_USER'),
                                  password=os.getenv('CH_PWD'))
    dt_list = []
    mhz_list = []
    zero_percent_list = []
    data = np.zeros((62128, 192, 4), dtype=np.complex128)
    bad_io = []
    for ms in msl:
        try:
            with table(ms, ack=False, readonly=True) as t:
            # get shape of DATA column
                if t.nrows() > data.shape[0]:
                    bad_io.append(ms)
                    continue
                t.getcolnp('DATA', data)
        except Exception as e:
            logger.error(f'Error in sanity check for {ms}: {e}')
            bad_io.append(ms)
            continue

        fn = os.path.basename(ms).rstrip('MHz.ms')
        date, time, mhz = fn.split('_')
        dt = datetime.strptime(f'{date}{time}', '%Y%m%d%H%M%S')
        zero_percent = int(100*(data == 0).sum() / data.size)
        zero_percent_list.append(zero_percent)
        dt_list.append(dt)
        mhz_list.append(int(mhz))
    client.insert('slowviz.zero_percent', [dt_list, mhz_list, zero_percent_list],
                  column_names=['timestamp', 'mhz', 'zero_percent'], column_oriented=True)
    return bad_io

def gainQA(
    calfile: str,
    do_plot: bool = True,
    save_stats: bool = True,
    outdir: str = './',
    qa_file: str = None,
):
    """
    Analyze a CASA gain calibration table to identify problematic antennas and channels
    based on gain amplitude and SNR statistics. Optionally generates a QA report and diagnostic plots.

    Parameters:
        calfile (str): Path to the CASA calibration table (gaincal output).
        do_plot (bool): If True, generates diagnostic plots and saves to `outdir`.
        save_stats (bool): If True, writes a .qa report with flagged antennas/channels.
        outdir (str): Output directory to save plots. Defaults to './'.
        qa_file (str): Optional path to the QA report file. If None, defaults to `calfile + '.qa'`.

    Returns:
        bad_corrs (np.ndarray): Array of indices of CORRELATOR NUMBERS with anomalous amplitude statistics.
        bad_ch (np.ndarray): Array of indices of channels with anomalous SNR statistics.
    """
    print("--> Analyzing ", calfile)

    # Load gain calibration table
    tb = tables.table(calfile, readonly=True, ack=False)
    gains = tb.getcol("CPARAM")  
    snr   = tb.getcol("SNR")     
    
    
    if "FLAG" in tb.colnames():
        flags = tb.getcol("FLAG")  # Boolean mask
    else:
        flags = np.zeros_like(gains, dtype=bool)
    tb.close()

    # Mask gains based on flags
    ma_gains = np.ma.masked_array(gains, mask=flags)
    ma_snr   = np.ma.masked_array(snr, mask=flags)

    # Compute amplitude from complex gain
    amp = np.abs(ma_gains)

    # Compute statistics
    amp_med_ant = np.ma.median(amp, axis=1)        
    snr_med_ch  = np.ma.median(ma_snr, axis=0)  
    amp_gmed    = np.ma.median(amp, axis=(0, 1))   
    amp_gstd    = np.ma.std(amp, axis=(0, 1))      
    snr_gmed    = np.ma.median(ma_snr, axis=(0, 1))
    snr_gstd    = np.ma.std(ma_snr, axis=(0, 1))   

    # Thresholds for flagging
    th_snr = 3
    th_amp = 4

    # Identify bad channels
    chid = np.arange(0, 192)
    bad_ch = np.asarray(chid)[np.ma.where(
        (snr_med_ch[:, 0] < snr_gmed[0] - th_snr * snr_gstd[0]) |
        (snr_med_ch[:, 1] < snr_gmed[1] - th_snr * snr_gstd[1])
    )]

    # Identify bad antennas
    antid = np.arange(0, 352)
    bad_ants = np.asarray(antid)[np.ma.where(
        (amp_med_ant[:, 0] < amp_gmed[0] - th_amp * amp_gstd[0]) |
        (amp_med_ant[:, 0] > amp_gmed[0] + th_amp * amp_gstd[0]) |
        (amp_med_ant[:, 1] < amp_gmed[1] - th_amp * amp_gstd[1]) |
        (amp_med_ant[:, 1] > amp_gmed[1] + th_amp * amp_gstd[1])
    )]

    # Write QA report
    if save_stats:
        statfile = qa_file if qa_file is not None else calfile + '.qa'
        with open(statfile, 'w') as f:
            print(f"Median gain amplitude (XX, YY): {amp_gmed}", file=f)
            print(f"Std of gain amplitude (XX, YY): {amp_gstd}", file=f)
            print(f"Median gain SNR (XX, YY)      : {snr_gmed}", file=f)
            print(f"Std of gain SNR (XX, YY)      : {snr_gstd}", file=f)
            print(f"AMP threshold : {th_amp}", file=f)
            print(f"SNR threshold : {th_snr}", file=f)

            if len(bad_ants) > 0:
                print(f"Antennas with Amp outside of range: {bad_ants}", file=f)
                for i in bad_ants:
                    print(f"Ant {i}, Med Amp (XX, YY):", amp_med_ant[i, :], file=f)
            else:
                print("All antennas have gain amp within range", file=f)

            if len(bad_ch) > 0:
                print(f"Channels with SNR outside of range: {bad_ch}", file=f)
                for i in bad_ch:
                    print(f"Ch {i}, Med SNR (XX, YY): {snr_med_ch[i, :]}", file=f)
            else:
                print("All channels have SNR within range", file=f)
        print(f"QA report saved to {statfile}")

    # Diagnostic plots
    if do_plot:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        xmin = np.amin(amp_med_ant) * 0.9
        xmax = np.amax(amp_med_ant) * 1.1
        nbins = int((xmax - xmin) / 0.00005)
        ax.hist(amp_med_ant[:, 0], bins=nbins, range=(xmin, xmax), label='XX',
                color='skyblue', edgecolor='black', alpha=0.3)
        ax.hist(amp_med_ant[:, 1], bins=nbins, range=(xmin, xmax), label='YY',
                color='red', edgecolor='black', alpha=0.3)
        ax.hist(amp_med_ant[bad_ants, 0], bins=nbins, range=(xmin, xmax),
                label=f'Bad ants:{bad_ants}', color='grey', edgecolor='black', alpha=1)
        ax.hist(amp_med_ant[bad_ants, 1], bins=nbins, range=(xmin, xmax),
                color='grey', edgecolor='black', alpha=1)

        ax.axvline(x=amp_gmed[0], c='skyblue')
        ax.axvline(x=amp_gmed[0] - th_amp * amp_gstd[0], c='skyblue', ls='--')
        ax.axvline(x=amp_gmed[0] + th_amp * amp_gstd[0], c='skyblue', ls='--')
        ax.axvline(x=amp_gmed[1], c='r')
        ax.axvline(x=amp_gmed[1] - th_amp * amp_gstd[1], c='r', ls='--')
        ax.axvline(x=amp_gmed[1] + th_amp * amp_gstd[1], c='r', ls='--')
        ax.set_xlabel("Median Amp gains")
        ax.set_ylabel("Frequency")
        ax.legend()
        fig.savefig(os.path.join(outdir, 'gains_AMP.png'), bbox_inches='tight')
        plt.close(fig)

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        xmin = np.amin(snr_med_ch) * 0.9
        xmax = np.amax(snr_med_ch) * 1.1
        nbins = int((xmax - xmin) / 0.5)
        ax.hist(snr_med_ch[:, 0], bins=nbins, range=(xmin, xmax), label='XX',
                color='skyblue', edgecolor='black', alpha=0.3)
        ax.hist(snr_med_ch[:, 1], bins=nbins, range=(xmin, xmax), label='YY',
                color='red', edgecolor='black', alpha=0.3)
        ax.hist(snr_med_ch[bad_ch, 0], bins=nbins, range=(xmin, xmax),
                label=f'Bad ch:{bad_ch}', color='grey', edgecolor='black', alpha=1)
        ax.hist(snr_med_ch[bad_ch, 1], bins=nbins, range=(xmin, xmax),
                color='grey', edgecolor='black', alpha=1)
        ax.axvline(x=snr_gmed[0], c='skyblue')
        ax.axvline(x=snr_gmed[0] - th_snr * snr_gstd[0], c='skyblue', ls='--')
        ax.axvline(x=snr_gmed[0] + th_snr * snr_gstd[0], c='skyblue', ls='--')
        ax.axvline(x=snr_gmed[1], c='r')
        ax.axvline(x=snr_gmed[1] - th_snr * snr_gstd[1], c='r', ls='--')
        ax.axvline(x=snr_gmed[1] + th_snr * snr_gstd[1], c='r', ls='--')
        ax.set_xlabel("SNR of gain solution per channel")
        ax.set_ylabel("Frequency")
        ax.legend()
        fig.savefig(os.path.join(outdir, 'gains_SNR.png'), bbox_inches='tight')
        print(f"QA plots saved to {outdir}")


    return bad_ants, bad_ch

