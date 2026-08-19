import pandas as pd
import random
import requests

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from data_scrapper import get_all_tickers, data_scrapper_unit
from data_filler import get_unix_timestamps
from data_preprocessor import calculate_additional_hyperparameters
from setup import API_KEY, TIMEFRAME, BAR_WIDTH

end = datetime.now(ZoneInfo("America/New_York")).date() + timedelta(1)
start = end - timedelta(2)
end = end.strftime("%Y-%m-%d")
start = start.strftime("%Y-%m-%d")

def get_random_ticker():
    """
    Returns a randomly selected ticker
    """
    return random.choice(get_all_tickers("raw_data/all_tickers", API_KEY))

def get_ticker_predata(ticker: str):
    url = f"https://api.massive.com/v2/aggs/ticker/{ticker}/range/{TIMEFRAME}/{start}/{end}?apiKey={API_KEY}"

    print(ticker)
    return data_scrapper_unit(url)

def fill_ticker_predata(df, date_range = pd.date_range(start, end), bar_width = BAR_WIDTH):
    expected_t = get_unix_timestamps(date_range, bar_width)
    expected = pd.DataFrame({"t": expected_t})

    df = expected.merge(df, on="t", how="left")

    df["f"] = df["c"].isna().astype(int)
    df["fb"] = 0

    df["date"] = (
        pd.to_datetime(df["t"], unit="ms", utc=True)
        .dt.tz_convert("America/New_York")
        .dt.date
    )

    min_t = df.loc[df["c"].notna(), "t"].min()
    max_t = df.loc[df["c"].notna(), "t"].max()

    df = df[(df["t"] >= min_t) &(df["t"] <= max_t)].copy()

    df["c_ff"] = df.groupby("date")["c"].ffill()
    for col in ["o", "h", "l", "c", "vw"]:
        df[col] = df[col].fillna(df["c_ff"])

    df.loc[df["f"] == 1, "v"] = 0
    df.loc[df["f"] == 1, "n"] = 0

    df["c_bf"] = df.groupby("date")["c"].bfill()
    backfill_mask = ((df["f"] == 1) & (df["c_ff"].isna()) & (df["c_bf"].notna()))
    df.loc[backfill_mask, "fb"] = 1
    for col in ["o", "h", "l", "c", "vw"]:
        df[col] = df[col].fillna(df["c_bf"])

    return (df.drop(columns=["c_ff", "c_bf"]).sort_values("t").reset_index(drop=True))

def process_ticker_predata(df):
    print(df.isna().sum().sum())

if __name__ == "__main__":
    tkr = get_random_ticker()
    valid, df, time = get_ticker_predata(tkr)
    process_ticker_predata(df)
    df = fill_ticker_predata(df)
    print(df)
