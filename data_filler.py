import os
import pandas as pd 
import duckdb
from tqdm import tqdm
import json

from pathlib import Path

from setup import DATE_RANGE, BAR_PER_DAY, BAR_WIDTH
from setup import path_data_scrapper, path_data_filler

def get_unix_timestamps(date_range: pd.DatetimeIndex, bar_width: int):
    return pd.to_datetime([
        f"{date} {time}"
        for date in date_range.strftime("%Y-%m-%d")
        for time in pd.date_range(
                        f"09:{30+bar_width}",
                        f"16:00",
                        freq=f"{bar_width}min"
                    ).strftime("%H:%M")
        ], format="%Y-%m-%d %H:%M").tz_localize("America/New_York").as_unit("ms").astype("int64")

def fill_data(input_folder, output_folder, date_range):
    """
    Corrects for missing intraday data
    """
    con = duckdb.connect()
    con.execute("SET memory_limit = '16GB'")
    con.execute("SET temp_directory = 'duckdb_temp'")

    #* Iterates through the raw data files
    batches_made = 0
    os.makedirs(output_folder, exist_ok=True)
    files = list(Path(input_folder).glob("*.parquet"))
    timestamps_df = pd.DataFrame({"t": get_unix_timestamps(date_range, BAR_WIDTH)})
    pbar = tqdm(files, total=len(files), desc=f"Setting up...".ljust(80),
                bar_format="|{bar}| {percentage:3.1f}% ({elapsed}) {desc}")
    for file_path in files:
        con.execute(f"""
            CREATE TEMP TABLE raw_data AS
            SELECT * EXCLUDE (T_1), 
                T_1 as Tk
            FROM read_parquet('{file_path}')
            ORDER BY Tk, t
        """)
        
        all_ticker_timestamps = con.sql(f"""
            SELECT
                *,
                CAST(
                    timezone(
                        'America/New_York',
                        to_timestamp(t / 1000.0)
                    )
                    AS DATE
                ) AS date,
                CASE
                    WHEN c IS NULL THEN 1
                    ELSE 0
                END AS f
            FROM (
                SELECT
                    tickers.Tk,
                    timestamps_df.t
                FROM (
                    SELECT DISTINCT Tk
                    FROM raw_data
                ) AS tickers
                CROSS JOIN timestamps_df
            ) AS expected
            LEFT JOIN raw_data
            USING (Tk, t)
        """)

        valid_days = con.sql("""
            SELECT *
            FROM all_ticker_timestamps
            QUALIFY MIN(f) OVER (
                PARTITION BY Tk, date
            ) = 0
        """)

        #* Assigns c_ff to all timestamps discluding backfill days
        forward_fill_step1 = con.sql(f"""
            SELECT
                *,
                LAST_VALUE(c IGNORE NULLS) OVER (
                    PARTITION BY Tk
                    ORDER BY t
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) as c_ff
            FROM valid_days
        """)
        #* Fills ohlc & vw to c_ff for ff days, sets v & n to 0 FOR ALL SYNTHETIC DAYS
        forward_fill_step2 = con.sql(f"""
            SELECT
                * REPLACE (
                    COALESCE(o, c_ff) as o,
                    COALESCE(h, c_ff) as h,
                    COALESCE(c, c_ff) as c,
                    COALESCE(l, c_ff) as l,
                    COALESCE(vw, c_ff) as vw,
                    CASE
                        WHEN f = 1 THEN 0
                        ELSE v
                    END AS v,
                    CASE
                        WHEN f = 1 THEN 0
                        ELSE n
                    END AS n
                )
            FROM forward_fill_step1
        """)

        pbar.set_description(f"Saving file for {str(file_path.stem)}...")
        con.execute(f"""
            COPY (
                SELECT * EXCLUDE (c_ff)
                FROM forward_fill_step2
                WHERE c IS NOT NULL
            )
            TO '{output_folder}/{file_path.stem}.parquet'
            (FORMAT PARQUET)
        """)
        con.execute("DROP TABLE raw_data")

def debug(source_path, condition = "True"):
    db = duckdb.sql(f"""
        SELECT *
        FROM read_parquet('{source_path}/batch_005.parquet')
        WHERE {condition}
        ORDER BY Tk, t
    """)
    return db

if __name__ == "__main__":
    with open("raw_data/all_tickers_trimmed_1_30", "r") as f:
        tickers = json.load(f)
    fill_data(path_data_scrapper, path_data_filler, DATE_RANGE)
    print(debug('filled_raw_data/data_5min_2025', "Tk = 'TENX'"))
