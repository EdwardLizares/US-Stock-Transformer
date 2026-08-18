import os
import pandas as pd 
import duckdb
from tqdm import tqdm
import json

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

def split_save(con, batch_num, train_end, train_path, val_end, val_path, test_path):
    con.execute(f"""
        COPY (
            SELECT * EXCLUDE (c_ff, c_bf)
            FROM batch
            WHERE date <= DATE '{train_end}'
        )
        TO '{train_path}/batch{batch_num}.parquet'
        (FORMAT PARQUET)
    """)
    con.execute(f"""
        COPY (
            SELECT * EXCLUDE (c_ff, c_bf)
            FROM batch
            WHERE date > DATE '{train_end}'
                AND date < DATE '{val_end}'
        )
        TO '{val_path}/batch{batch_num}.parquet'
        (FORMAT PARQUET)
    """)
    con.execute(f"""
        COPY (
            SELECT * EXCLUDE (c_ff, c_bf)
            FROM batch
            WHERE date >= DATE '{val_end}'
        )
        TO '{test_path}/batch{batch_num}.parquet'
        (FORMAT PARQUET)
    """)

def impute_data(input_path, output_path, tickers, date_range, split = [0.75, 0.9]):
    """
    Corrects for missing intraday data and creates train-val-test splits by date with duckdb.
    """
    con = duckdb.connect()
    con.execute("SET memory_limit = '16GB'")
    con.execute("SET temp_directory = 'duckdb_temp'")

    dates = date_range.date
    train_idx = int(len(dates) * 0.75)
    val_idx = int(len(dates) * 0.90)

    train_end = dates[train_idx]
    val_end = dates[val_idx]

    train_path = f"{output_path}/train"
    val_path = f"{output_path}/val"
    test_path = f"{output_path}/test"
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    
    #* Creates a table of all the data
    con.execute(f"""
        CREATE TEMP TABLE raw_data AS
        SELECT * EXCLUDE (otc)
        FROM read_parquet('{input_path}')
    """)

    batches_made, batch_exists, batch_limit, batch_count = 0, False, 1000, 0
    timestamps_df = pd.DataFrame({"t": get_unix_timestamps(DATE_RANGE, BAR_WIDTH)})
    pbar = tqdm(tickers, total=len(tickers), desc=f"Sending jobs to threads...".ljust(80),
                bar_format="|{bar}| {percentage:3.1f}% ({elapsed}) {desc}")
    os.makedirs(output_path, exist_ok=True)
    for ticker in pbar:
        pbar.set_description(f"Filling data for {ticker}...")
        #* Gets the cur tkr table
        tkr_db = con.sql(f"""
            SELECT *
            FROM raw_data
            WHERE T_1 = '{ticker}'
        """)

        #* Fills all possible days
        all_days = con.sql(f"""
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
                END AS f,
                0 AS fb
            FROM (
                SELECT
                    '{ticker}' AS T_1,
                    timestamps_df.t
                FROM timestamps_df
            ) AS expected

            LEFT JOIN tkr_db
            USING (T_1, t)
        """)

        #* Removes invalid days (before real start, after real close)
        valid_days = con.sql(f"""
            SELECT *
            FROM all_days
            WHERE 
                t >= (
                    SELECT MIN(t)
                    FROM tkr_db
                )
                AND
                t <= (
                    SELECT MAX(t)
                    FROM tkr_db
                )
        """)

        #* Assigns c_ff to all timestamps discluding backfill days
        forward_fill_step1 = con.sql(f"""
            SELECT 
            *,
            LAST_VALUE(c IGNORE NULLS) OVER (
                PARTITION BY date
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

        #* Assigns c_bf to all backfill days
        back_fill_step1 = con.sql("""
            SELECT
                *,
                FIRST_VALUE(c IGNORE NULLS) OVER (
                    PARTITION BY date
                    ORDER BY t
                    ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING
                ) AS c_bf
            FROM forward_fill_step2
        """)

        #* Fills ohlc & vw to c_bf for bf days
        back_fill_step2 = con.sql("""
            SELECT
                * REPLACE (
                    CASE
                        WHEN f = 1
                            AND c_ff IS NULL
                            AND c_bf IS NOT NULL
                        THEN 1
                        ELSE 0
                    END AS fb,

                    COALESCE(o,  c_bf) AS o,
                    COALESCE(h,  c_bf) AS h,
                    COALESCE(l,  c_bf) AS l,
                    COALESCE(c,  c_bf) AS c,
                    COALESCE(vw, c_bf) AS vw,
                )
            FROM back_fill_step1
            ORDER BY t
        """)
        batch_count+=1
        if not batch_exists:
            con.execute("""
                CREATE TEMP TABLE batch AS
                SELECT * FROM back_fill_step2
            """)
            batch_exists = True
        else:
            con.execute("""
                INSERT INTO batch
                SELECT * FROM back_fill_step2
            """)
        if batch_count >= batch_limit:
            split_save(con, batches_made, train_end, train_path, val_end, val_path, test_path)
            con.execute("""DROP TABLE batch""")
            batch_exists = False
            batches_made += 1
            batch_count = 0
    if batch_exists:
        split_save(con, batches_made, train_end, train_path, val_end, val_path, test_path)

def debug(source_path):
    db = duckdb.sql(f"""
        SELECT *
        FROM read_parquet('{source_path}/batch0.parquet')
    """)
    return db

if __name__ == "__main__":
    with open("raw_data/all_tickers_trimmed_1_30", "r") as f:
        tickers = json.load(f)
    impute_data(path_data_scrapper, path_data_filler, tickers, DATE_RANGE)
    #print(debug(path_data_filler))
