import requests
import json
import pandas as pd
import time
import pandas_market_calendars as mcal

from itertools import product
from tqdm import tqdm

API_KEY = "sdpbiDy3nfhuvQX2SBBtL6Gt2dl88ZrU"
DATE_RANGE = mcal.get_calendar("NYSE").schedule("2025-01-01","2025-12-31").index
MN = 1
MX = 30
file_path = "test/test_data/15min_bars"

def get_all_tickers() -> list[str]:
    try:
        with open("raw_data/all_tickers", "r") as f:
            return json.load(f)
    except FileNotFoundError as _:
        tickers = []
        url = f"https://api.massive.com/v3/reference/tickers?market=stocks&active=true&limit=1000&apiKey={API_KEY}"

        while url:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            resp = response.json()

            tickers.extend([r["ticker"] for r in resp.get("results", [])])

            next_url = resp.get("next_url")
            url = f"{next_url}&apiKey={API_KEY}" if next_url else None

        with open("raw_data/all_tickers", "w", encoding='utf-8') as f:
            json.dump(tickers, f)
            return tickers
        
def filter_ticker_list_by_price_range(ticker_list: list[str], date_range: pd.DatetimeIndex, 
                            mn: int, mx: int) -> list[str]:
    """
    Trims tickers that have never been within the given price range (mn-mx)
    """
    try:
        with open("processed_data/all_tickers_trimmed", "r") as f:
            return json.load(f)
    except FileNotFoundError as _:
        pass

    f, s, n = 0, 0, len(date_range)
    rqt, ct = 0, 0 # tracks time for requests and calculations
    pbar = tqdm(enumerate(date_range), total=n,
                bar_format="|{bar}| {percentage:3.1f}% ({elapsed}) {desc}")
    trimmed_ticker_set = set()
    for i, date in pbar:
        dstr = date.strftime("%Y-%m-%d")
        pbar.set_description_str((f"[{f}/{s}/{n}] "
                                  f"{{{rqt/(i+1):.1f}|{ct/(i+1):.1f}}} "
                                  f"-{len(trimmed_ticker_set)}-"
                                  f"Requesting data for {dstr}".ljust(60)))
        url = f"https://api.massive.com/v2/aggs/grouped/locale/us/market/stocks/{dstr}?apiKey={API_KEY}"
        try:
            t0=time.time()
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            resp = response.json()
            t1=time.time()
            for row in resp["results"]:
                if row["T"] in trimmed_ticker_set:
                    continue
                else:
                    if ((mn < row["o"] and mx > row["o"]) or
                        (mn < row["c"] and mx > row["c"])):
                        trimmed_ticker_set.add(row["T"])
            t2=time.time()
            rqt += t1-t0
            ct += t2-t1
            s += 1
        except requests.exceptions.RequestException as _:
            f += 1
            pass
        else:
            pass

    trimmed_tickers = list(trimmed_ticker_set)
    with open("processed_data/all_tickers_trimmed", "w", encoding='utf-8') as f:
        json.dump(trimmed_tickers, f)
        return trimmed_tickers

def data_scrapper(ticker_list, date_range: pd.DatetimeIndex,
                  bar_width: str) -> pd.DataFrame:
    """
    Iterates through all ticker_list x data_range combinations given bar_width (str)
    Returns a dataframe of all the raw data scraped from Massive
    """
    s, f, rqt, pt, n = 0, 0, 0, 0, len(ticker_list)
    errors = []
    data = pd.DataFrame()
    d_stt, d_end = date_range[[0,-1]].strftime("%Y-%m-%d")
    pbar = tqdm(enumerate(ticker_list), total=n,
                bar_format="|{bar}| {percentage:3.1f}% ({elapsed}) {desc}")
    for i, tkr in pbar:
        pbar.set_description_str((f"[{f}/{s}/{n}] "
                                  f"{{{rqt/(i+1):.1f}|{pt/(i+1):.1f}}} "
                                  f"Fetching bars for {tkr} from {d_stt}-{d_end}".ljust(60)))
        url = (f"https://api.massive.com/v2/aggs/ticker/{tkr}/range/{bar_width}/"
                f"{d_stt}/{d_end}?adjusted=true&sort=asc&limit=50000&apiKey={API_KEY}")
        try:
            t0 = time.time()
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            t1 = time.time()
            resp = response.json()
            t2 = time.time()
            rqt += t1-t0
            pt += t2-t1
        except requests.exceptions.RequestException as _:
            f += 1

        s += 1
        df = pd.DataFrame(resp["results"])
        data = pd.concat([data, df], ignore_index=True)

    pbar.set_description_str("Data fetching completed")
    return data

if __name__ == "__main__":
    all_tickers = get_all_tickers()
    print(len(all_tickers))
    all_tickers_trimmed = filter_ticker_list_by_price_range(all_tickers, DATE_RANGE, MN, MX)
    print(len(all_tickers_trimmed))
    data_scrapper(all_tickers_trimmed, DATE_RANGE, "15/minute").to_parquet(f"{file_path}.parquet", index=False)