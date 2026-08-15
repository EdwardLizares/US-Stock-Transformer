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

def data_to_df(tickers, unix_timestamps):
    tickers_df = pd.DataFrame({
        "T": ["AAPL", "MSFT", "NVDA"]
    })
    timestamps_df = pd.DataFrame({
        "t": get_unix_timestamps(DATE_RANGE, BAR_WIDTH)
    })
    return tickers_df, timestamps_df

def get_expected_t():
    """
    Returns the expected ticker, date, t_ms combos as an sql string
    """
    return f"""
        SELECT
            ticker.T,
            times.bar_time
        FROM tickers_df ticker
        CROSS JOIN timestamps_df
    """

def forward_back_fill(source_path: str):
    duckdb.sql("""
        SELECT *
        FROM 'source_path.parquet'
        
    """)
    return

def test_db(source_path: str):
    test = duckdb.sql(f"""
        SELECT *
        FROM '{source_path}'
        GROUP BY T_1, t
        LIMIT 10
    """)
    goal = duckdb.sql(f"""
        SELECT *
        FROM  '{"filled_raw_data/data_15min_2025.parquet"}'
        LIMIT 10
    """)
    return test, goal

def duckdb_fill():
    con = duckdb.connect()
    con.execute("""SET enable_progress_bar = true""")
    step1 = con.sql(f"""
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
            SELECT tickers_df.T_1, timestamps_df.t
            FROM tickers_df
            CROSS JOIN timestamps_df
        ) AS expected
        LEFT JOIN read_parquet('{path_data_scrapper}') AS data
        USING (T_1, t)
        ORDER BY T_1, t
        LIMIT 10000
    """)
    ff = con.sql(f"""
        SELECT 
        *,
        LAST_VALUE(c IGNORE NULLS) OVER (
            PARTITION BY T_1, date
            ORDER BY t
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) as c_ff
        FROM step1
        ORDER BY T_1, t
        LIMIT 10000
    """)
    ff = con.sql(f"""
        SELECT
        * REPLACE (
            COALESCE(o, c_ff) as o,
            COALESCE(c, c_ff) as c,
            COALESCE(h, c_ff) as h,
            COALESCE(l, c_ff) as l,
            COALESCE(vw, c_ff) as vw,
            CASE
                WHEN f = 1 AND c_ff IS NOT NULL THEN 0
                ELSE v
            END AS v,
            CASE
                WHEN f = 1 AND c_ff IS NOT NULL THEN 0
                ELSE n
            END AS n
        )
        FROM ff
        ORDER BY T_1, t
        LIMIT 10000
    """)
    bf = con.sql("""
        SELECT
            *,
            FIRST_VALUE(c IGNORE NULLS) OVER (
                PARTITION BY T_1, date
                ORDER BY t
                ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING
            ) AS c_bf
        FROM ff
    """)
    bf = con.sql("""
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
                COALESCE(c,  c_bf) AS c,
                COALESCE(h,  c_bf) AS h,
                COALESCE(l,  c_bf) AS l,
                COALESCE(vw, c_bf) AS vw,

                CASE
                    WHEN f = 1
                        AND c_ff IS NULL
                        AND c_bf IS NOT NULL
                    THEN 0
                    ELSE v
                END AS v,

                CASE
                    WHEN f = 1
                        AND c_ff IS NULL
                        AND c_bf IS NOT NULL
                    THEN 0
                    ELSE n
                END AS n
            )
        FROM bf
        ORDER BY T_1, t
        LIMIT 10000
    """)
    return bf
    
if __name__ == "__main__":
    with open("raw_data/all_tickers_trimmed_1_30", "r") as f:
        tickers = json.load(f)
    tickers_df = pd.DataFrame({"T_1": tickers})
    timestamps_df = pd.DataFrame({"t": get_unix_timestamps(DATE_RANGE, BAR_WIDTH)})
    duckdb_fill() # Accesses tickers_df and timestamps_df internally
    print(bf)