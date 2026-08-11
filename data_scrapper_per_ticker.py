import requests
import pandas as pd
import time
from tqdm import tqdm
import pandas_market_calendars as mcal

API_KEY = "sdpbiDy3nfhuvQX2SBBtL6Gt2dl88ZrU"
DATA_RANGE = mcal.get_calendar("NYSE").schedule("2025-01-01","2025-12-31").index

def data_scrapper(api_key: str, date_range: pd.DatetimeIndex) -> dict:
    s, f = 0, 0
    progress_bar = tqdm(enumerate(date_range), total=len(date_range),
                        bar_format="|{bar}| {percentage:3.1f}% ({elapsed}) {desc}")
    data = pd.DataFrame()
    errors = []
    request_time = []
    json_parse_time = []
    for i, date in progress_bar:
        date_str = date.strftime("%Y-%m-%d")
        progress_bar.set_description_str(f"[{f}/{s}/{len(date_range)}] Fetching data for {date_str}".ljust(60))
        url = (f"https://api.massive.com/v2/aggs/grouped/"
               f"locale/us/market/stocks/{date_str}?apiKey={api_key}")
        try:
            t0 = time.time()
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            t1 = time.time()
            resp = response.json()
            t2 = time.time()
            request_time.append(t1-t0)
            json_parse_time.append(t2-t1)
        except requests.exceptions.RequestException as e:
            errors.append((date_str, str(e)))
            f += 1
            progress_bar.set_description_str(
                f"[{f}/{s}/{len(date_range)}] "
                f"Request failed for {date_str}:".ljust(60)
            )

        s += 1
        df = pd.DataFrame(resp["results"])[["T", "v", "vw", "o", "c", "h", "l", "n"]]
        df = df.rename(columns={"T": "Ticker", "v": "Volume", "vw": "VWAP", "o": "Open", "c": "Close",
                                "h": "High", "l": "Low", "n": "NumTrades"})
        df["Date"] = date_str
        data = pd.concat([data, df], ignore_index=True)

    progress_bar.set_description_str("Data fetching completed")
    time.sleep(0.5)

    print(f"Request time ~{request_time.mean()} | Parse time ~{json_parse_time.mean()}")
    return data

if __name__ == "__main__":
    data_scrapper(API_KEY, DATA_RANGE).to_parquet("raw_data/data_day_2024.parquet", index=False)