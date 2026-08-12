import pandas as pd
import pandas_market_calendars as mcal

from data_processor import get_unix_timestamps

from setup import SEQ_LEN
from setup import path_data_processor, path_dataset_builder

def trim_data(df: pd.DataFrame):
    return df

if __name__ == "__main__":
    print("Reading source path parquet...")
    df = pd.read_parquet(path_data_processor)
    print("Culling empty days...")
    df = df[(df["o"].notna())] #* Should be removing whole days if data processed correctly
    print("Searching for days that meet rv requirements...")


    print(df)