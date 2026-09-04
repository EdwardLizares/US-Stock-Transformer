import os
import pandas as pd

from pathlib import Path
from datetime import timedelta

from setup import API_KEY, TIMEFRAME
from data_scrapper import range_scrapper_unit
from data_filler import fill_file
from data_preprocessor import preprocess_file

def handpick_raw_data(api_key, output_folder, ticker, date):
    """
    Fetches data for a single ticker on a single date
    """
    os.makedirs(output_folder, exist_ok=True)
    date_str = date.strftime("%Y-%m-%d")
    success, df, rqt_time = range_scrapper_unit(api_key, ticker, pd.date_range(date - timedelta(110), date), TIMEFRAME)
    if not success:
        print(f"Failed to fetch data for {ticker} on {date_str}")
        return
    if df.empty:
        print(f"No data found for {ticker} on {date_str}")
        return
    df["T"] = ticker
    batch_path = os.path.join(output_folder, f"{ticker}_{date_str}.parquet")
    df.to_parquet(batch_path, index=False)
    print(f"Data for {ticker} on {date_str} saved to {batch_path} in {rqt_time:.2f}s")

def handpick_data_filler(input_file, output_folder, date):
    os.makedirs(output_folder, exist_ok=True)
    fill_file(Path(input_file), output_folder, pd.date_range(date - timedelta(110), date), store=True)

def handpicked_data_preprocessor(input_file, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    preprocess_file(Path(input_file), Path(output_folder), split=[0,0], split_names=["test"])

def handpick_data():
    ticker = input("Enter ticker: ").upper()
    date = input("Enter date (YYYY-MM-DD): ")

    handpicked_raw_data_folder = f"handpicked_data/raw_data"
    handpick_raw_data(API_KEY, handpicked_raw_data_folder, ticker, pd.to_datetime(date))

    handpicked_raw_data_path = f"handpicked_data/raw_data/{ticker}_{date}.parquet"
    handpicked_filled_data_folder = f"handpicked_data/filled_data"
    handpick_data_filler(handpicked_raw_data_path, handpicked_filled_data_folder, pd.to_datetime(date))

    handpicked_filled_data_path = f"handpicked_data/filled_data/{ticker}_{date}.parquet"
    handpicked_preprocessed_data_folder = f"handpicked_data/preprocessed_data"
    handpicked_data_preprocessor(handpicked_filled_data_path, handpicked_preprocessed_data_folder)
    
if __name__ == "__main__":
    #print(pd.read_parquet("handpicked_data/raw_data/RDHL_2026-08-31.parquet"))
    handpick_data()