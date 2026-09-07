import os
import pandas as pd 
import duckdb
from tqdm import tqdm
import json

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from data_preprocessor import preprocess_data

from setup import DATE_RANGE, BAR_WIDTH
from setup import path_data_scrapper, path_data_filler, path_data_preprocessor

def get_unix_timestamps(date_range: pd.DatetimeIndex, bar_width: int):
    return pd.to_datetime([
        f"{date} {time}"
        for date in date_range.strftime("%Y-%m-%d")
        for time in pd.date_range(
                        f"08:{59}",
                        f"15:59", 
                        freq=f"{1}min"
                    ).strftime("%H:%M")
        ], format="%Y-%m-%d %H:%M").tz_localize("America/New_York").as_unit("ms").astype("int64")

def fill_file(file_path, output_folder, date_range, store=True):
    """
    Corrects for missing intraday data for one file
    """
    con = duckdb.connect()
    con.execute("SET threads = 2")
    con.execute("SET memory_limit = '4GB'")
    temp_directory = Path("duckdb_temp") / file_path.stem
    temp_directory.mkdir(parents=True, exist_ok=True)

    con.execute(f"SET temp_directory = '{temp_directory.as_posix()}'")
    timestamps_df = pd.DataFrame({"t": get_unix_timestamps(date_range, 1)})
    
    con.execute(f"""
        CREATE TEMP TABLE raw_data AS
        SELECT * EXCLUDE (T_1), 
            T_1 as Tk
        FROM read_parquet('{file_path.as_posix()}')
        ORDER BY Tk, t
    """)

    previous_available_bar = con.sql("""
        WITH days AS (
            SELECT DISTINCT
                Tk,
                CAST(timezone('America/New_York', to_timestamp(t / 1000.0)) AS DATE) AS date
            FROM raw_data
        ),
        seed_times AS (
            SELECT
                Tk,
                date,
                epoch_ms(
                    timezone(
                        'America/New_York',
                        CAST(date AS TIMESTAMP) + INTERVAL '8 hours 59 minutes'
                    )
                ) AS seed_t
            FROM days
        )
        SELECT
            s.Tk,
            s.date,
            s.seed_t AS t,
            r.o,
            r.h,
            r.l,
            r.c,
            r.vw
        FROM seed_times s
        ASOF LEFT JOIN raw_data r
            ON s.Tk = r.Tk
            AND s.seed_t >= r.t
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

    all_ticker_timestamps = con.sql("""
        SELECT
            a.* REPLACE (
                COALESCE(a.o, s.o) AS o,
                COALESCE(a.h, s.h) AS h,
                COALESCE(a.l, s.l) AS l,
                COALESCE(a.c, s.c) AS c,
                COALESCE(a.vw, s.vw) AS vw
            )
        FROM all_ticker_timestamps a
        LEFT JOIN previous_available_bar s
            ON a.Tk = s.Tk
            AND a.date = s.date
            AND a.t = s.t
    """)

    valid_days = con.sql("""
        SELECT *
        FROM all_ticker_timestamps
        QUALIFY MIN(f) OVER (
            PARTITION BY Tk, date
        ) = 0
    """)

    forward_fill_step1 = con.sql(f"""
        SELECT
            *,
            LAST_VALUE(c IGNORE NULLS) OVER (
                PARTITION BY Tk, date
                ORDER BY t
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) as c_ff
        FROM valid_days
    """)

    forward_fill_step2 = con.sql(f"""
        SELECT
            * REPLACE (
                COALESCE(o, c_ff) as o,
                COALESCE(h, c_ff) as h,
                COALESCE(c, c_ff) as c,
                COALESCE(l, c_ff) as l,
                COALESCE(vw, c_ff) as vw,
                CASE WHEN f = 1 THEN 0 ELSE v END AS v,
                CASE WHEN f = 1 THEN 0 ELSE n END AS n
            )
        FROM forward_fill_step1
        QUALIFY COUNT_IF(c_ff IS NULL) OVER (
            PARTITION BY Tk, date
        ) = 0
    """)

    final_data = con.sql("""
        SELECT *
        FROM forward_fill_step2
        WHERE NOT (
            EXTRACT(hour FROM timezone('America/New_York', to_timestamp(t / 1000.0))) = 8
            AND EXTRACT(minute FROM timezone('America/New_York', to_timestamp(t / 1000.0))) = 59
        )
    """)

    output_path = Path(output_folder) / f"{file_path.stem}.parquet"
    con.execute(f"""
        COPY (
            SELECT * EXCLUDE (c_ff)
            FROM final_data
            ORDER BY Tk, t
        )
        TO '{output_path.as_posix()}'
        (FORMAT PARQUET)
    """)
    con.execute("DROP TABLE raw_data")
    con.close()

    if not store:
        #* Processes only this worker's file
        preprocess_data(output_folder, path_data_preprocessor, date_range, False, file_path=output_path)
        output_path.unlink()
    return file_path.stem

def fill_data(input_folder, output_folder, date_range, store=True, workers=3):
    """
    Corrects for missing intraday data
    """
    os.makedirs(output_folder, exist_ok=True)
    files = list(Path(input_folder).glob("*.parquet"))
    pbar = tqdm(total=len(files), desc=f"Setting up...".ljust(80),
                bar_format="|{bar}| {percentage:3.1f}% ({elapsed}) {desc}")
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(fill_file, file_path, output_folder, date_range, store): file_path
            for file_path in files
        }
        for future in as_completed(futures):
            file_path = futures[future]
            try:
                batch = future.result()
                pbar.set_description(f"Completed {batch}...".ljust(80))
            except Exception as e:
                pbar.write(f"Error processing {file_path}: {e}")
                raise
            pbar.update(1)
    pbar.close()

def debug():
    db = duckdb.sql("""
        SELECT
            Tk,
            MIN(date) AS first_date,
            MAX(date) AS last_date,
            COUNT(*) AS rows
        FROM read_parquet(
            'filled_raw_data/data_1min_2023_2025/batch_000.parquet'
        )
        WHERE Tk = 'AEF'
        GROUP BY Tk
    """)

    print(db)

if __name__ == "__main__":
    with open("raw_data/all_tickers_trimmed_1_30", "r") as f:
        tickers = json.load(f)
    fill_data(path_data_scrapper, path_data_filler, DATE_RANGE, True)
    print(pd.read_parquet("filled_raw_data/data_1min_2021_2026/batch_001.parquet"))
