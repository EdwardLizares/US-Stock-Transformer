import torch
import requests
import pandas as pd

from time import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from data_filler import get_unix_timestamps
from data_preprocessor import calculate_additional_hyperparameters, engineer_data

def live_scrapper_unit(url: str, max_retries: int = 2):
    for attempt in range(max_retries + 1):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            resp = response.json()
            return pd.DataFrame(resp.get("results", []))
        except requests.RequestException:
            if attempt < max_retries:
                time.sleep(attempt + 1)
            else:
                raise

def get_ticker_predata(app_state, ticker: str):
    url = (f"https://api.massive.com/v2/aggs/ticker/{ticker}"
           f"/range/{app_state.get_timeframe()}/"
           f"{(app_state.date-timedelta(days=2))}/{app_state.date}?"
           f"apiKey={app_state.get_apikey()}")
    df = live_scrapper_unit(url)
    df = df.drop(columns=["otc"], errors="ignore")
    return df

def fill_ticker_predata(app_state, ticker, df):
    now = datetime.now(ZoneInfo("America/New_York"))
    expected_t = get_unix_timestamps(
        pd.date_range(app_state.date - timedelta(days=2), app_state.date), app_state.bar_width
    )
    now_ms = int(now.timestamp() * 1000)
    expected_t = expected_t[expected_t <= now_ms]

    df = df.set_index("t").reindex(expected_t).rename_axis("t").reset_index()
    df["date"] = (pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert("America/New_York").dt.date)

    #* Forward fill
    f_mask = df['c'].isna()
    df["f"] = f_mask.astype(int)
    df.loc[f_mask, ['v','n']] = 0
    c_ff = df['c'].ffill()
    for col in ["o", "h", "l", "c", "vw"]:
        df[col] = df[col].fillna(c_ff)
    df = df[df["c"].notna()].reset_index(drop=True)    
    df["Tk"] = ticker
    #print(df)
    return df

def process_ticker_predata(app_state, df):
    df = calculate_additional_hyperparameters(df)
    df = engineer_data(df)
    df = df[app_state.get_inputfeatures() + ['Tk', 'date']]
    df = df.iloc[:(-15//app_state.bar_width)]
    df = df.tail(90)
    #print(df)
    return df

def add_ticker_predata(app_state, output_path, ticker):
    raw = get_ticker_predata(app_state, ticker)
    df = fill_ticker_predata(app_state, ticker, raw)
    df = process_ticker_predata(app_state, df)

    return df
