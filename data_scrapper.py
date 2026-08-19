import requests
import json
import pandas as pd
import time

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from setup import API_KEY, DATE_RANGE, TIMEFRAME, MN, MX, BAR_WIDTH
from setup import path_raw_all_tickers, path_raw_tickers_trimmed, path_data_scrapper

def get_all_tickers(path: str, api_key: str) -> list[str]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        tickers = []

        for ticker_type in ("CS", "ETF"):
            url = (
                "https://api.massive.com/v3/reference/tickers"
                f"?market=stocks"
                f"&type={ticker_type}"
                f"&active=true"
                f"&limit=1000"
                f"&apiKey={api_key}"
            )

            while url:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                resp = response.json()

                tickers.extend([
                    r["ticker"] for r in resp.get("results", [])
                ])

                next_url = resp.get("next_url")
                url = f"{next_url}&apiKey={api_key}" if next_url else None

        with open(path, "w", encoding="utf-8") as f:
            json.dump(tickers, f)

        print(f"CS + ETFs: {len(tickers)}")
        return tickers

def gen_aggregate_url(api_key: str, date: str):
    return f"https://api.massive.com/v2/aggs/grouped/locale/us/market/stocks/{date}?apiKey={api_key}"

def gen_range_url(api_key: str, ticker: str, time_frame: str, d_stt, d_end):
    return (f"https://api.massive.com/v2/aggs/ticker/{ticker}/range/{time_frame}/"
           f"{d_stt}/{d_end}?adjusted=true&sort=asc&limit=50000&apiKey={api_key}")

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

def aggregate_scrapper_unit(api_key: str, date: str) -> tuple[bool, pd.DataFrame, int]:
    return data_scrapper_unit(gen_aggregate_url(api_key, date))

def range_scrapper_unit(api_key: str, tkr: str, date_range: pd.DatetimeIndex, time_frame: str) -> tuple[bool, pd.DataFrame, int]:
    return data_scrapper_unit(gen_range_url(api_key, tkr, time_frame, date_range[0].strftime("%Y-%m-%d"), date_range[-1].strftime("%Y-%m-%d")))

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
        thread_jobs = {executor.submit(aggregate_scrapper_unit, API_KEY, dstr.strftime("%Y-%m-%d")): dstr.strftime("%Y-%m-%d") for dstr in date_range}
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

def data_scrapper(api_key:str, output_path: str, 
                  ticker_list: list, date_range: pd.DatetimeIndex, time_frame: str) -> pd.DataFrame:
    """
    Returns a DataFrame of all the raw data scraped from Massive
    Creates a .parquet of the DataFrame on first run
    """
    try:
        return pd.read_parquet(output_path)
    except FileNotFoundError as _:
        pass
    s, f, avg_rqt, n = 0, 0, 0, len(ticker_list)
    dfs = []
    d_stt, d_end = date_range[[0,-1]].strftime("%Y-%m-%d")
    pbar = tqdm(ticker_list, total=n, desc="Sending jobs to threads...".ljust(80),
                bar_format="|{bar}| {percentage:3.1f}% ({elapsed}) {desc}")
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        thread_jobs = {executor.submit(range_scrapper_unit, api_key,
                                       tkr, date_range, time_frame): tkr for tkr in ticker_list}
        for job in as_completed(thread_jobs): # --> runs as jobs are completed
            tkr = thread_jobs[job]
            res = job.result() # (Success? DF, rqt time, pt_time)
            s += res[0]
            f += (not res[0])
            if not res[1].empty:
                res[1]["T"] = tkr
                dfs.append(res[1])
            pbar.update(1)
            avg_rqt = avg_rqt + (res[2]-avg_rqt)/pbar.n
            pbar.set_description_str((f"[{f}/{s}/{n}] "
                                      f"{{{avg_rqt:.1f}/r/t}} "
                                      f"OHLC bars for {tkr} from {d_stt}-{d_end} added".ljust(80)))
    data = pd.concat(dfs, ignore_index=True)
    pbar.set_description_str("Data fetching completed")
    data.to_parquet(f"{output_path}", index=False)
    return data
import requests
import pandas as pd

if __name__ == "__main__":
    all_tickers = get_all_tickers(path_raw_all_tickers, API_KEY)
    print(len(all_tickers))
    all_tickers_trimmed = filter_ticker_list_by_price_range(path_raw_tickers_trimmed,
                                                            DATE_RANGE,
                                                            MN,
                                                            MX)
    print(len(all_tickers_trimmed))
    ds = data_scrapper(
        api_key=API_KEY,
        output_path=path_data_scrapper,
        ticker_list=all_tickers_trimmed,
        date_range=DATE_RANGE,
        time_frame=TIMEFRAME,
    )
    print(ds)
