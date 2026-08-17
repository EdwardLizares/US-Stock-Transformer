import os
import duckdb
import pandas as pd

from tqdm import tqdm
from pathlib import Path 

from setup import AVG_VOLUME_PERIOD, RV_THRESH, MN, MX, BAR_PER_DAY
from setup import path_data_filler, path_data_preprocessor

class ProcessingError(Exception):
    pass

def calculate_additional_hyperparameters(df: pd.DataFrame, pbar) -> pd.DataFrame:
    pbar.set_description("Doing some feature engineering...".ljust(80))
    df = df.sort_values(["T_1", "t"])
    df["av"] = (
        df.groupby("T_1")["v"].transform(
            lambda x: x.rolling(AVG_VOLUME_PERIOD).mean()
            )
        )
    df["ema9"] = (
        df.groupby("T_1")["c"].transform(
            lambda x: x.ewm(span=9, adjust=False).mean()
        )
    )
    df["ema20"] = (
        df.groupby("T_1")["c"].transform(
            lambda x: x.ewm(span=20, adjust=False).mean()
        )
    )
    df["ema12"] = (
        df.groupby("T_1")["c"].transform(
            lambda x: x.ewm(span=12, adjust=False).mean()
        )
    )
    df["ema26"] = (
        df.groupby("T_1")["c"].transform(
            lambda x: x.ewm(span=26, adjust=False).mean()
        )
    )
    df["macd"] = (df["ema12"]-df["ema26"])
    p_c = df.groupby("T_1")["c"].shift(1)
    df["gp"] = (df["o"] - p_c) / p_c * 100
    df["cp"] = (df["c"] - df["o"]) / df["c"] * 100
    df["rv"] = df["v"] / df["av"]

    #n_o = df.groupby("T_1")["o"].shift(-1)
    #df["y"] = (n_o > df["c"]).astype(int)

    df["bar"] = df.groupby(["T_1", "date"]).cumcount() + 1
    df.insert(df.columns.get_loc("date") + 1, "bar", df.pop("bar"))

    return df

def engineer_data(df: pd.DataFrame, pbar) -> pd.DataFrame:
    df = calculate_additional_hyperparameters(df, pbar)
    return df

def filter_data(df: pd.DataFrame, pbar) -> pd.DataFrame:
    """
    Filters for RV and Price
    Additionally removes days with any NaN rv
    """
    pbar.set_description("Filtering data...".ljust(80))
    rv_mask = (df.groupby(["T_1", "date"])["rv"].transform("max")>RV_THRESH)
    df = df[rv_mask]
    pc_mask = ((df.groupby(["T_1", "date"])["l"].transform("min")<=MX) &
               (df.groupby(["T_1", "date"])["h"].transform("min")>MN))
    df = df[pc_mask]

    nan_mask = (df.groupby(["T_1", "date"])["rv"].transform("count") < BAR_PER_DAY)
    df = df[~nan_mask]

    return df

def preprocess_data(source_folder: str, output_folder: str):
    """
    Processes .parquet files in a given source_folder
    Creates a folder of .parquet files
    """
    os.makedirs(output_folder, exist_ok=True)
    files = list(Path(source_folder).glob("*.parquet"))
    pbar = tqdm(files, f"Setting up...".ljust(80),
                bar_format="|{bar}| {percentage:3.1f}% ({elapsed}) {desc}")
    for filled_batch_path in pbar:
        if Path(f"{output_folder}/{filled_batch_path.stem}.parquet").exists() and True:
            pbar.write(f"{str(filled_batch_path)} has already been preprocessed...")
            continue #* Doesn't recalculate
        pbar.set_description(f"Reading source path parquet at {str(filled_batch_path)}...".ljust(80))
        df = pd.read_parquet(filled_batch_path)

        df = engineer_data(df, pbar)
        df = filter_data(df, pbar)
        df = df.sort_values(["date", "T_1", "bar"])       #! IMPORTANT FOR LATER TEST_VAL_SPLIT
        df = df.set_index(["T_1", "date"])

        float_cols = df.select_dtypes(include=["float64"]).columns
        df[float_cols] = df[float_cols].astype("float32")

        output_path = f"{output_folder}/{filled_batch_path.stem}"
        pbar.set_description(f"Saving data to {output_path}...".ljust(80))
        df.to_parquet(f"{output_path}.parquet", index=True)
    pbar.set_description("Data preprocessing complete")

def debug(source_path, columns: str = '*'):
    db = duckdb.sql(f"""
        SELECT {columns}
        FROM read_parquet('{source_path}/batch0.parquet')
    """)
    return db

if __name__ == "__main__":
    #preprocess_data(path_data_filler, path_data_preprocessor)
    print(debug(path_data_preprocessor, "f, fb"))
    
