from datetime import date
import pandas as pd

FLAG_TABLE = '/opt/devel/yuping/ant_flags.csv'
df = pd.read_csv(FLAG_TABLE, parse_dates=['date'])

def get_bad_ants(date: date, sources=['AI-VAR', 'AI-LO']):
    """Return a list of bad antennas for a given date and kinds of flags.
    
    Args:
        date: The date
        sources: Sources of flags. Default is ['AI-VAR', 'AI-LO'].

    Returns:
        A list of bad antenna corr numbers.
    """
    res = df[df['source'].isin(sources) & (df['date'] == str(date))]['corr_num'].sort_values().unique().tolist()
    if not res or len(res) == 0:
        raise ValueError(f"No bad antennas found for {date}.")
    return res