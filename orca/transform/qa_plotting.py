from matplotlib.backends.backend_pdf import PdfPages
from casatools import table
import numpy as np
import matplotlib.pyplot as plt
import os


def plot_bandpass_to_pdf_amp_phase(calfile, pdf_path="bandpass_QA_all.pdf", msfile=None):
    """
    Generate a multi-page PDF visualizing bandpass calibration solutions per antenna.

    Each antenna is shown with two vertically stacked subplots:
    - Amplitude (log scale) vs frequency
    - Phase (in degrees) vs frequency

    Parameters
    ----------
    calfile : str
        Path to the CASA bandpass calibration table (e.g. '.bandpass').
        Must contain the 'CPARAM' column.
    pdf_path : str, optional
        Path to the output PDF file (default: 'bandpass_QA_all.pdf').
    msfile : str, optional
        Optional path to the associated measurement set ('.ms').
        Used to extract frequency values in MHz from the SPECTRAL_WINDOW table.
        If unavailable, channel indices are used instead.

    Notes
    -----
    - Each page shows 16 antennas (4 columns × 4 rows).
    - Flagged data points are shown in red.
    - Amplitude plots use log scale with fixed limits [1e-4, 2e0] if all unflagged values fit.
    - Phase plots are fixed to [-180, 180] degrees.
    - Legends show Pol 0, Pol 1, and Flagged in every amplitude subplot.
    """


    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)

    tb = table()
    tb.open(calfile)
    gains = tb.getcol("CPARAM")
    npol, nchan, nant = gains.shape

    if "FLAG" in tb.colnames():
        flags = tb.getcol("FLAG")
    else:
        flags = np.zeros_like(gains, dtype=bool)

    if "ANTENNA1" in tb.colnames():
        antennas = tb.getcol("ANTENNA1")
    else:
        antennas = np.arange(nant)
    tb.close()

    # Get frequency axis from MS if available
    freq = None
    x_label = "Frequency index"
    if msfile is not None and os.path.exists(msfile):
        try:
            tb_ms = table()
            tb_ms.open(os.path.join(msfile, "SPECTRAL_WINDOW"))
            freq = np.array(tb_ms.getcell("CHAN_FREQ", 0)) / 1e6  # MHz
            tb_ms.close()
            x_label = "Frequency (MHz)"
        except Exception as e:
            print(f"Warning: failed to read MS freq: {e}")
    if freq is None or len(freq) != nchan:
        freq = np.arange(nchan)

    with PdfPages(pdf_path) as pdf:
        print(f"Generating PDF: {pdf_path}")
        ncols = 4
        antennas_per_page = 16
        total_pages = (nant + antennas_per_page - 1) // antennas_per_page

        for page in range(total_pages):
            fig, axs = plt.subplots(8, ncols, figsize=(20, 15), sharex='col')
            axs = np.array(axs).reshape(8, ncols)

            print(f"Page {page+1}/{total_pages}")
            for i in range(antennas_per_page):
                ant_global_idx = page * antennas_per_page + i
                if ant_global_idx >= nant:
                    break

                col = i % ncols
                row_amp = (i // ncols) * 2
                row_phase = row_amp + 1

                ax_amp = axs[row_amp, col]
                ax_phase = axs[row_phase, col]

                handles = []
                labels = []

                for pol in range(npol):
                    gain = gains[pol, :, ant_global_idx]
                    flag = flags[pol, :, ant_global_idx]
                    amp = np.abs(gain)
                    phase = np.degrees(np.angle(gain))
                    color = f'C{pol}'

                    # Amplitude
                    l1, = ax_amp.plot(freq[~flag], amp[~flag], '-o', markersize=2, color=color)
                    ax_amp.plot(freq[flag], amp[flag], 'o', markersize=2, color='red')

                    # Phase
                    ax_phase.plot(freq[~flag], phase[~flag], '-o', markersize=2, color=color)
                    ax_phase.plot(freq[flag], phase[flag], 'o', markersize=2, color='red')

                    handles.append(l1)
                    labels.append(f'Pol {pol}')

                # Flagged handle
                red_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=5)
                handles.append(red_patch)
                labels.append("Flagged")

                # Amplitude formatting
                ax_amp.set_yscale("log")
                valid_amps = np.abs(gains[:, :, ant_global_idx])[~flags[:, :, ant_global_idx]]
                if valid_amps.size > 0 and np.all((valid_amps >= 1e-4) & (valid_amps <= 2e0)):
                    ax_amp.set_ylim(1e-4, 2e0)
                ax_amp.set_title(f"Antenna {antennas[ant_global_idx]}")
                ax_amp.grid(True)

                # Phase formatting
                ax_phase.set_ylim(-180, 180)
                ax_phase.grid(True)

                if row_phase == 7:
                    ax_phase.set_xlabel(x_label)
                if col == 0:
                    ax_amp.set_ylabel("Amp")
                    ax_phase.set_ylabel("Phase (deg)")

                ax_amp.legend(handles=handles, labels=labels, fontsize='small', loc='upper right')

                print(f"  → Plotted antenna {antennas[ant_global_idx]} [{i+1}/16 on page {page+1}]")

            # Hide unused axes
            for r in range(8):
                for c in range(ncols):
                    idx = page * antennas_per_page + (r // 2) * ncols + c
                    if idx >= nant:
                        axs[r, c].remove()

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()
            print(f"Saved page {page+1} of PDF.")



def plot_delay_vs_antenna(delay_table_path, output_pdf="delay_vs_antenna.pdf"):
    """
    Plot delay (in ns) vs antenna from CASA .delay table.
    Flagged points shown in red. Y-axis limited to [-10000, 10000] ns if all data fits.
    Output saved as a PDF.
    """
    tb = table()
    tb.open(delay_table_path)

    delays = tb.getcol("FPARAM")        # shape: (npol, 1, nant)
    flags = tb.getcol("FLAG")           # same shape
    antenna_ids = tb.getcol("ANTENNA1") # shape: (nant,)
    tb.close()

    delays_ns = delays[:, 0, :] #in ns
    flags = flags[:, 0, :]
    n_pol, n_ant = delays_ns.shape

    colors = ['blue', 'orange']
    legend_done = {"Pol 0": False, "Pol 1": False, "Flagged": False}

    # Check unflagged values for limit enforcement
    valid_delays = delays_ns[~flags]
    use_fixed_limits = np.all((valid_delays >= -10000) & (valid_delays <= 10000))

    plt.figure(figsize=(12, 6))
    for pol in range(n_pol):
        for ant in range(n_ant):
            delay_val = delays_ns[pol, ant]
            if flags[pol, ant]:
                label = "Flagged" if not legend_done["Flagged"] else None
                plt.plot(antenna_ids[ant], delay_val, 'o', color='red', label=label)
                legend_done["Flagged"] = True
            else:
                label = f"Pol {pol}" if not legend_done[f"Pol {pol}"] else None
                plt.plot(antenna_ids[ant], delay_val, 'o', color=colors[pol], label=label)
                legend_done[f"Pol {pol}"] = True

    plt.xlabel("Antenna")
    plt.ylabel("Delay (ns)")
    plt.title("Delay per Antenna")
    if use_fixed_limits:
        plt.ylim(-10000, 10000)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_pdf)
    plt.close()
    print(f"Saved PDF to: {output_pdf}")

