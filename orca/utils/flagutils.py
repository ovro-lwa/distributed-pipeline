"""Flag file utilities for antenna and visibility flagging.

Provides functions for reading and writing flag data, looking up
bad antennas from the flag database, and visualizing flag statistics.
"""
import datetime
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import subprocess
import ast
import textwrap

FLAG_TABLE = '/opt/devel/yuping/ant_flags.csv'

# Load flag table if it exists, otherwise create empty DataFrame
if os.path.exists(FLAG_TABLE):
    df = pd.read_csv(FLAG_TABLE, parse_dates=['date'])
else:
    df = pd.DataFrame(columns=['date', 'ant', 'source'])

def get_bad_ants(date: datetime.date, sources=['AI-VAR', 'AI-LO']):
    """Return a list of bad antennas for a given date and kinds of flags.
    
    Args:
        date: The date
        sources: Sources of flags. Default is ['AI-VAR', 'AI-LO'].

    Returns:
        A list of bad antenna corr numbers.
    """
    # Sigh, hardcoded gaps.
    if date == datetime.date(2024, 1, 23):
        date = datetime.date(2024, 1, 22)
    elif datetime.date(2024, 5, 7) < date <= datetime.date(2024, 5, 11):
        date = datetime.date(2024, 5, 7)
    elif datetime.date(2024, 5, 11) < date < datetime.date(2024, 5, 15):
        date = datetime.date(2024, 5, 15)
    elif date > datetime.date(2024, 5, 20): # last available date
        date = datetime.date(2024, 5, 20)
    res = df[df['source'].isin(sources) & (df['date'] == str(date))]['corr_num'].sort_values().unique().tolist()
    if not res or len(res) == 0:
        raise ValueError(f"No bad antennas found for {date}.")
    return res

def unpack_flag_metadata(input_file: str, original_shape: tuple) -> np.ndarray:
    """
    Unpacks binary flag metadata from a packed binary file into its original FLAG column shape.

    Parameters:
    -----------
    input_file : str
        Path to the binary file containing the packed flag data.
    original_shape : tuple
        The shape of the original FLAG array (e.g., (polarizations, channels, rows)).

    Returns:
    --------
    np.ndarray
        A Boolean array of the same shape as the original FLAG column.
    """
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"The file '{input_file}' does not exist.")

    print(f"Loading packed flags from '{input_file}'...")
    bit_packed = np.fromfile(input_file, dtype=np.uint8)

    # Calculate the total number of bits that need to be unpacked
    total_bits = np.prod(original_shape)
    total_bytes = (total_bits + 7) // 8  # Calculate how many bytes are needed

    if bit_packed.size < total_bytes:
        raise ValueError(f"Insufficient data in the file. Expected at least {total_bytes} bytes, got {bit_packed.size}.")

    print("Unpacking flags...")
    flags = np.unpackbits(bit_packed)[:total_bits].astype(bool)  # Only unpack the required number of bits
    flags = flags.reshape(original_shape)

    print(f"Unpacked FLAG column shape: {flags.shape}")
    print(f"Total flagged points: {np.sum(flags)}")

    return flags

def plot_flag_metadata_all_polarizations_subplot(flags: np.ndarray, output_dir: str = None) -> None:
    """
    Plots the flag metadata for all polarizations in a single subplot figure.

    Parameters:
    -----------
    flags : np.ndarray
        A Boolean array of shape (polarizations, channels, rows) representing the flag data.
    output_dir : str, optional
        Directory where the plot will be saved. If None, the plot is displayed interactively.
    """
    if flags.ndim != 3:
        raise ValueError(f"Expected a 3D FLAG array (polarizations, channels, rows), but got shape {flags.shape}.")

    num_polarizations = flags.shape[0]

    fig, axes = plt.subplots(1, num_polarizations, figsize=(20, 4)) 

    for pol in range(num_polarizations):
        ax = axes[pol] if num_polarizations > 1 else axes
        im = ax.imshow(flags[pol], aspect='auto', cmap='viridis', origin='lower')
        ax.set_title(f'Polarization {pol}')
        ax.set_xlabel('Row Index')
        if pol == 0:
            ax.set_ylabel('Channel Index')

    fig.colorbar(im, ax=axes.ravel().tolist(), label='Flagged (True/False)')

    if output_dir:
        output_path = os.path.join(output_dir, 'flag_metadata_all_polarizations.png')
        plt.savefig(output_path)
        print(f"Saved subplot for all polarizations to '{output_path}'")
    else:
        plt.show()

    plt.close()


# Test the function on the provided file
# unpacked_flags = unpack_flag_metadata(input_file, original_shape)
# plot_flag_metadata_all_polarizations_subplot(unpacked_flags, output_dir=None)  # Set output_dir to save the plot instead of displaying it


def get_bad_antenna_names(date_time_str):
    """
    Get a list of bad antenna names (e.g., 'LWA-005B') using the 'development' conda environment.

    Args:
        date_time_str (str): Date and time string in ISO format, e.g. '2025-01-28 19:20:04'

    Returns:
        list: List of bad antenna names
    """
    code = textwrap.dedent("""
        from mnc import anthealth
        from astropy.time import Time
        dt = '{}'
        b = anthealth.get_badants('selfcorr', time=Time(dt, format='iso').mjd)
        print(b[1])
    """.format(date_time_str))

    result = subprocess.run(
        ["conda", "run", "-n", "development", "python", "-c", code],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        raise RuntimeError("Error running command:\n{}".format(result.stderr))

    lines = result.stdout.strip().splitlines()
    list_line = next((line for line in reversed(lines) if line.startswith("['LWA-")), None)

    if not list_line:
        raise ValueError("Could not find a valid list of bad antennas in output:\n{}".format(result.stdout))

    try:
        return ast.literal_eval(list_line)
    except Exception as e:
        raise ValueError("Failed to parse antenna list:\n{}".format(list_line)) from e


def get_bad_correlator_numbers(date_time_str):
    code = textwrap.dedent("""
        from mnc import anthealth
        from astropy.time import Time
        import lwa_antpos.mapping as mapping

        dt = '{}'
        b = anthealth.get_badants('selfcorr', time=Time(dt, format='iso').mjd)
        badnames = b[1]
        correlators = [
            mapping.antname_to_correlator(name.rstrip('A').rstrip('B'))
            for name in badnames
        ]
        print(sorted(set(correlators)))
    """.format(date_time_str))

    result = subprocess.run(
        ["conda", "run", "-n", "development", "python", "-c", code],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        raise RuntimeError("Error running command:\n{}".format(result.stderr))

    lines = result.stdout.strip().splitlines()
    list_line = next((line for line in reversed(lines) if line.startswith("[")), None)

    if not list_line:
        raise ValueError("Could not find a valid list of correlators in output:\n{}".format(result.stdout))

    try:
        return ast.literal_eval(list_line)
    except Exception as e:
        raise ValueError("Failed to parse correlator list:\n{}".format(list_line)) from e
