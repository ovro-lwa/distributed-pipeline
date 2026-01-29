"""Cleanup utility for measurement sets with excessive zero values.

Queries ClickHouse database for measurement sets with high zero-value
percentages and removes them from disk. Used for data quality maintenance.
"""
import clickhouse_connect
import os
import shutil


def main():
    """Remove measurement sets with >10% zero values from disk.

    Queries the slowviz.zero_percent table in ClickHouse and deletes
    corresponding directories from the pipeline storage.
    """
    client = clickhouse_connect.get_client(host='10.41.0.85', username=os.getenv('CH_USER'),
                                  password=os.getenv('CH_PWD'))
    res = client.query('SELECT timestamp, mhz FROM slowviz.zero_percent WHERE zero_percent > 10 AND toMonth(timestamp) = 6')
    counts = 0
    for r in res.result_rows:
        ts = r[0]
        mhz = r[1]
        dir = f"/lustre/pipeline/night-time/{mhz}MHz/{ts.date()}/{ts.hour:02d}/{ts.strftime('%Y%m%d_%H%M%S')}_{mhz}MHz.ms"
        if os.path.exists(dir):
            shutil.rmtree(dir)
            counts += 1
            print(f"Removed {dir}")
        else:
            print(f"{dir} not found.")
    print(f"Removed {counts} directories.")

if __name__ == '__main__':
    main()