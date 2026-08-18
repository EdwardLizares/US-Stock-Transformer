import os
import pyarrow as pa
import pandas as pd

from tqdm import tqdm
from pathlib import Path 

from setup import AVG_VOLUME_PERIOD, RV_THRESH, MN, MX, BAR_PER_DAY, INPUT_FEATURES
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
    Takes a folder with train-val-test subfolders 
    Creates a folder of .parquet files
    """
    assert {p.name for p in Path(source_folder).glob("*/")} == {"test", "train", "val"}, "Incorrect source folder format"
    os.makedirs(output_folder, exist_ok=True)
    folders = list(Path(source_folder).glob("*/"))
    file_paths = []
    for folder in folders:
        paths = Path(folder).glob("*.parquet")
        file_paths += [p for p in paths]

    pbar = tqdm(file_paths, f"Setting up...".ljust(80),
                bar_format="|{bar}| {percentage:3.1f}% ({elapsed}) {desc}")
    for imputed_batch_path in pbar:
        if Path(f"{output_folder}/{imputed_batch_path.stem}.arrow").exists():
            pbar.write(f"{str(imputed_batch_path)} has already been preprocessed...")
            continue #* Doesn't recalculate
        pbar.set_description(f"Reading source path parquet at {str(imputed_batch_path)}...".ljust(80))
        df = pd.read_parquet(imputed_batch_path)

        df = engineer_data(df, pbar)
        df = filter_data(df, pbar)
        df = df.sort_values(["date", "T_1", "bar"])
        df = df[INPUT_FEATURES+["T_1", "date"]]

        float_cols = df.select_dtypes(include=["float64"]).columns
        df[float_cols] = df[float_cols].astype("float32")

        #* Convert to arrow
        table = pa.Table.from_pandas(df,preserve_index=False)
        output_path = Path(output_folder) / imputed_batch_path.relative_to(source_folder)
        output_path = output_path.with_suffix(".arrow")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if Path(output_path).exists():
            pbar.set_description("File cannot be overwritten!")
            continue
        else:
            pbar.set_description(f"Saving data to {output_path}...".ljust(80))

        with pa.OSFile(str(output_path), "wb") as sink:
            with pa.ipc.new_file(sink, table.schema) as writer:
                writer.write_table(table)
    pbar.set_description("Data preprocessing complete")

def debug():
    path = "preprocessed_data/data_5min_2025/train/batch0.arrow"
    with pa.memory_map(path, "r") as source:
        reader = pa.ipc.open_file(source)
        table = reader.read_all()

    print(table)

if __name__ == "__main__":
    preprocess_data(path_data_filler, path_data_preprocessor)
    print(debug())
    
