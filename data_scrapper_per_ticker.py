import requests
import json
import pandas as pd
import time
import pandas_market_calendars as mcal

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

API_KEY = "sdpbiDy3nfhuvQX2SBBtL6Gt2dl88ZrU"
DATE_RANGE = mcal.get_calendar("NYSE").schedule("2025-01-01","2025-12-31").index
TIMEFRAME = "1/hour"
MN = 1
MX = 30
FILE_PATH_GET_ALL_TICKERS = "raw_data/all_tickers"
file_path_filtered_ticker = "raw_data/all_tickers_trimmed_1_30"
file_path_data_scrapper = "raw_data/data_1hour_2025.parquet"

def get_all_tickers(path: str) -> list[str]:
    try:
        with open(path, "r") as f:
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

        with open(path, "w", encoding='utf-8') as f:
            json.dump(tickers, f)
            return tickers

def gen_aggregate_url(date: str):
    return f"https://api.massive.com/v2/aggs/grouped/locale/us/market/stocks/{date}?apiKey={API_KEY}"

def gen_range_url(ticker: str, bar_width: str, d_stt, d_end):
    return (f"https://api.massive.com/v2/aggs/ticker/{ticker}/range/{bar_width}/"
           f"{d_stt}/{d_end}?adjusted=true&sort=asc&limit=50000&apiKey={API_KEY}")

def data_scrapper_unit(url: str, max_retries: int = 2):
    t0, t1 = time.time(), 0
    for attempt in range(max_retries + 1):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            resp = response.json()
            t1 = time.time()
            if "results" not in resp or not resp["results"]:
                return (True, pd.DataFrame(), t1 - t0)
            return (True, pd.DataFrame(resp["results"]), t1 - t0)
        except Exception as _:
            if attempt < max_retries:
                time.sleep(1 * (attempt + 1))  # brief backoff before retrying
                continue
            t_fail = time.time()
            return (False, pd.DataFrame(), t_fail - t0)

def aggregate_scrapper_unit(date: str) -> tuple[bool, pd.DataFrame, int]:
    return data_scrapper_unit(gen_aggregate_url(date))

def range_scrapper_unit(tkr: str, date_range: pd.DatetimeIndex, bar_width: str) -> tuple[bool, pd.DataFrame, int]:
    return data_scrapper_unit(gen_range_url(tkr, bar_width, date_range[0].strftime("%Y-%m-%d"), date_range[-1].strftime("%Y-%m-%d")))

def filter_ticker_list_by_price_range(path: str, date_range: pd.DatetimeIndex,
                                      mn: int, mx: int) -> list[str]:
    """
    Trims tickers that have never been within the given price range (mn-mx)
    """
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError as _:
        pass

    s, f, avg_rqt, n = 0, 0, 0, len(date_range)
    data = pd.DataFrame()
    pbar = tqdm(date_range, total=n, desc=f"Sending jobs to threads...".ljust(80),
                bar_format="|{bar}| {percentage:3.1f}% ({elapsed}) {desc}")
    with ThreadPoolExecutor(max_workers=20) as executor:
        thread_jobs = {executor.submit(aggregate_scrapper_unit, dstr.strftime("%Y-%m-%d")): dstr.strftime("%Y-%m-%d") for dstr in date_range}
        for job in as_completed(thread_jobs): # --> runs as jobs are completed
            dstr = thread_jobs[job]
            res = job.result() # (Success? DF, rqt time, pt_time)
            s += res[0]
            f += (not res[0])
            data = pd.concat([data, res[1]], ignore_index=True)
            pbar.update(1)
            avg_rqt = avg_rqt + (res[2]-avg_rqt)/pbar.n
            pbar.set_description_str((f"[{f}/{s}/{n}] "
                                      f"{{{avg_rqt:.1f}/r/t}} "
                                      f"Requesting data for {dstr}".ljust(80)))
    trimmed_tickers = list(data[(((mn<data["o"]) & (mx>data["o"])) |
                                ((mn<data["c"]) & (mx>data["c"])))]["T"].unique())
    with open(path, "w", encoding='utf-8') as f:
        json.dump(trimmed_tickers, f)
        return trimmed_tickers

def data_scrapper(path: str, ticker_list, date_range: pd.DatetimeIndex,
                  bar_width: str) -> pd.DataFrame:
    """
    Iterates through all ticker_list x data_range combinations given bar_width (str)
    Returns a dataframe of all the raw data scraped from Massive
    """
    try:
        return pd.read_parquet(path)
    except FileNotFoundError as _:
        pass
    s, f, avg_rqt, n = 0, 0, 0, len(ticker_list)
    data = pd.DataFrame()
    d_stt, d_end = date_range[[0,-1]].strftime("%Y-%m-%d")
    pbar = tqdm(ticker_list, total=n, desc="Sending jobs to threads...".ljust(80),
                bar_format="|{bar}| {percentage:3.1f}% ({elapsed}) {desc}")
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        thread_jobs = {executor.submit(range_scrapper_unit,
                                       tkr, date_range, bar_width): tkr for tkr in ticker_list}
        for job in as_completed(thread_jobs): # --> runs as jobs are completed
            tkr = thread_jobs[job]
            res = job.result() # (Success? DF, rqt time, pt_time)
            s += res[0]
            f += (not res[0])
            if not res[1].empty:
                res[1]["T"] = tkr
                data = pd.concat([data, res[1]], ignore_index=True)
            pbar.update(1)
            avg_rqt = avg_rqt + (res[2]-avg_rqt)/pbar.n
            pbar.set_description_str((f"[{f}/{s}/{n}] "
                                      f"{{{avg_rqt:.1f}/r/t}} "
                                      f"OHLC bars for {tkr} from {d_stt}-{d_end} added".ljust(80)))

    pbar.set_description_str("Data fetching completed")
    return data

if __name__ == "__main__":
    all_tickers = get_all_tickers(FILE_PATH_GET_ALL_TICKERS)
    print(len(all_tickers))
    all_tickers_trimmed = filter_ticker_list_by_price_range(file_path_filtered_ticker,
                                                            DATE_RANGE,
                                                            MN,
                                                            MX)
    print(len(all_tickers_trimmed))
    ds = data_scrapper(file_path_data_scrapper, 
                       all_tickers_trimmed, 
                       DATE_RANGE,
                       TIMEFRAME)
    print(ds)
    ds.to_parquet(f"{file_path_data_scrapper}", index=False)
