import os
import pyarrow as pa
import pandas as pd

from tqdm import tqdm
from pathlib import Path 

from setup import AVG_VOLUME_PERIOD, RV_THRESH, MN, MX, BAR_PER_DAY, INPUT_FEATURES, DATE_RANGE, SPLIT
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

def preprocess_data(source_folder: str, output_folder: str, date_range, split):
    """
    Takes a folder with raw parquet files and splits each folder by date into train-val-test folders
    """
    #* GET TRAIN/VAL/TEST DATE BOUNDARIES
    dates = date_range.date

    train_idx = int(len(dates) * split[0])
    val_idx = int(len(dates) * split[1])
    train_end = dates[train_idx]
    val_end = dates[val_idx]

    #* CREATE OUTPUT FOLDERS
    output_folder = Path(output_folder)
    for split_name in ["train", "val", "test"]:
        (output_folder / split_name).mkdir(parents=True, exist_ok=True)

    file_paths = sorted(Path(source_folder).glob("*.parquet"))
    pbar = tqdm(file_paths, f"Setting up...".ljust(80),
                bar_format="|{bar}| {percentage:3.1f}% ({elapsed}) {desc}")
    for imputed_batch_path in pbar:
        output_paths = {
            split_name: output_folder/ split_name / imputed_batch_path.with_suffix(".arrow").name
            for split_name in ["train", "val", "test"]
        }
        if all(path.exists() for path in output_paths.values()):
            pbar.write(f"{imputed_batch_path} has already been preprocessed...")
            continue

        pbar.set_description(f"Reading source path parquet at {str(imputed_batch_path)}...".ljust(80))
        df = pd.read_parquet(imputed_batch_path)

        df = engineer_data(df, pbar)
        df = filter_data(df, pbar)
        df = df.sort_values(["date", "T_1", "bar"])
        df = df[INPUT_FEATURES+["T_1", "date"]]

        float_cols = df.select_dtypes(include=["float64"]).columns
        df[float_cols] = df[float_cols].astype("float32")

        split_dfs = {"train": df[df["date"] <= train_end],
                     "val": df[(df["date"] > train_end) & (df["date"] < val_end)],
                     "test": df[df["date"] >= val_end]}

        for split_name, split_df in split_dfs.items():
            output_path = output_paths[split_name]
            if output_path.exists():
                pbar.write(f"{output_path} already exists...")
                continue
            pbar.set_description(f"Saving {output_path}...".ljust(80))
            table = pa.Table.from_pandas(split_df, preserve_index=False)

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
    preprocess_data(path_data_filler, path_data_preprocessor, DATE_RANGE, SPLIT)
    print(debug())
    
