import os
from posixpath import split
import pyarrow as pa
import pandas as pd

from tqdm import tqdm
from pathlib import Path  
from concurrent.futures import ProcessPoolExecutor, as_completed

from setup import AVG_VOLUME_PERIOD, RV_THRESH, MN, MX, BAR_PER_DAY, INPUT_FEATURES, DATE_RANGE, SPLIT, DEBUG
from setup import path_data_filler, path_data_preprocessor

class ProcessingError(Exception):
    pass

def calculate_additional_hyperparameters(df: pd.DataFrame, pbar = None) -> pd.DataFrame:
    if pbar is not None:
        pbar.set_description("Doing some feature engineering...".ljust(80))
    df = df.sort_values(["Tk", "t"])
    df["av"] = (
        df.groupby("Tk")["v"].transform(
            lambda x: x.rolling(AVG_VOLUME_PERIOD).mean()
            )
        )
    df["ema9"] = (
        df.groupby("Tk")["c"].transform(
            lambda x: x.ewm(span=9, adjust=False).mean()
        )
    )
    df["ema20"] = (
        df.groupby("Tk")["c"].transform(
            lambda x: x.ewm(span=20, adjust=False).mean()
        )
    )
    df["ema12"] = (
        df.groupby("Tk")["c"].transform(
            lambda x: x.ewm(span=12, adjust=False).mean()
        )
    )
    df["ema26"] = (
        df.groupby("Tk")["c"].transform(
            lambda x: x.ewm(span=26, adjust=False).mean()
        )
    )
    df["macd"] = (df["ema12"]-df["ema26"])
    df["rv"] = df["v"] / df["av"]

    #n_o = df.groupby("T_1")["o"].shift(-1)
    #df["y"] = (n_o > df["c"]).astype(int)

    df["bar"] = df.groupby(["Tk", "date"]).cumcount() + 1
    df.insert(df.columns.get_loc("date") + 1, "bar", df.pop("bar"))

    return df

def calculate_ibkr_rv(df: pd.DataFrame, pbar=None) -> pd.DataFrame:
    if pbar is not None:
        pbar.set_description("Calculating IBKR RV...".ljust(80))

    df = df.sort_values(["Tk", "t"])
    daily_volume = (
        df.groupby(["Tk", "date"])["v"].sum()
        .rename("daily_v").reset_index()
    )
    daily_volume = daily_volume.sort_values(["Tk", "date"])
    daily_volume["avg_daily_v"] = (
        daily_volume.groupby("Tk")["daily_v"].transform(
            lambda x: x.shift(1).rolling(90).mean()
        )
    )

    df = df.merge(daily_volume[["Tk", "date", "avg_daily_v"]], 
                  on=["Tk", "date"], how="left")
    
    df["cum_v"] = (df.groupby(["Tk", "date"])["v"].cumsum())
    df["ibkr_rv"] = (df["cum_v"] / df["avg_daily_v"])
    return df

def engineer_data(df: pd.DataFrame, pbar = None) -> pd.DataFrame:
    df = calculate_additional_hyperparameters(df, pbar)
    df = calculate_ibkr_rv(df, pbar)
    return df

def filter_data(df: pd.DataFrame, pbar = None) -> pd.DataFrame:
    """
    Filters data
    Current: 10% range, Price 1-20, Rv>tresh
    """
    if pbar is not None:
        pbar.set_description("Filtering data...".ljust(80))

    #* Range filter
    day_low = df.groupby(["Tk", "date"])["l"].transform("min")
    day_high = df.groupby(["Tk", "date"])["h"].transform("max")
    range_mask = ((day_high - day_low) / day_low) >= 0.10
    df = df[range_mask]

    #* Price filter
    pc_mask = ((df.groupby(["Tk", "date"])["l"].transform("min")<=MX) &
               (df.groupby(["Tk", "date"])["h"].transform("min")>=MN))
    df = df[pc_mask]

    nan_mask = (df.groupby(["Tk", "date"])["rv"].transform("count") < BAR_PER_DAY)
    df = df[~nan_mask]

    #* Rv filter
    print("Before RV filter:",
        df.groupby(["Tk", "date"]).ngroups,
        len(df))

    ibkr_nan_mask = (df.groupby(["Tk", "date"])["ibkr_rv"].transform("count")
                     < BAR_PER_DAY)
    df = df[~ibkr_nan_mask]
    ibkr_rv_mask = (
        df.groupby(["Tk", "date"])["ibkr_rv"].transform("max")
        >= RV_THRESH
    )
    df = df[ibkr_rv_mask]

    print("After RV filter:",
        df.groupby(["Tk", "date"]).ngroups,
        len(df))

    return df

def preprocess_file(file_path, output_folder, split, split_names, train_end = None, val_end = None):
    output_paths = {split_name: output_folder/ split_name / file_path.with_suffix(".arrow").name
                    for split_name in split_names}
    if all(path.exists() for path in output_paths.values()):
        return

    df = pd.read_parquet(file_path)
    df = engineer_data(df, None)
    df = filter_data(df, None)
    df = df.sort_values(["date", "Tk", "bar"])
    df = df[INPUT_FEATURES+["Tk", "date", "ibkr_rv"]]

    float_cols = df.select_dtypes(include=["float64"]).columns
    df[float_cols] = df[float_cols].astype("float32")

    if split == [0,0]:
        split_dfs = {"test": df}
    else:
        split_dfs = {"train": df[df["date"] <= train_end],
                    "val": df[(df["date"] > train_end) & (df["date"] < val_end)],
                    "test": df[df["date"] >= val_end]}

    for split_name, split_df in split_dfs.items():
        output_path = output_paths[split_name]
        if output_path.exists():
            continue
        table = pa.Table.from_pandas(split_df, preserve_index=False)
        with pa.OSFile(str(output_path), "wb") as sink:
            with pa.ipc.new_file(sink, table.schema) as writer:
                writer.write_table(table)

def preprocess_data(source_folder, output_folder, date_range, set_pbar=True, split=[0.75, 0.9], file_path=None):    
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
    split_names = ["train", "val", "test"]
    if split == [0,0]:
        split_names = ["test"]
    for split_name in split_names:
        (output_folder / split_name).mkdir(parents=True, exist_ok=True)

    file_paths = sorted(Path(source_folder).glob("*.parquet")) if file_path is None else [Path(file_path)]
    pbar = tqdm(file_paths, f"Setting up...".ljust(80),
                bar_format="|{bar}| {percentage:3.1f}% ({elapsed}) {desc}") if set_pbar else None

    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(preprocess_file, file_path, output_folder, split, split_names, train_end, val_end): file_path
            for file_path in file_paths
        }
        for future in as_completed(futures):
            file_path = futures[future]
            try:
                future.result()
                if pbar is not None:
                    pbar.update(1)
                    pbar.write(f"Completed {file_path.name}...".ljust(80))
            except Exception as e:
                if pbar is not None:
                    pbar.write(f"Error processing {file_path}: {e}")
                raise

    if pbar is not None:
        if DEBUG:
            pbar.set_description("Data preprocessing complete")
    
def debug():
    path = "preprocessed_data/data_5min_2025/train/batch0.arrow"
    with pa.memory_map(path, "r") as source:
        reader = pa.ipc.open_file(source)
        table = reader.read_all()

    print(table)

def refilter_arrow_files(source_folder, output_folder):
    source_folder = Path(source_folder)
    output_folder = Path(output_folder)

    for split_name in ["train", "val", "test"]:
        split_source = source_folder / split_name
        split_output = output_folder / split_name
        split_output.mkdir(parents=True, exist_ok=True)

        files = sorted(split_source.glob("*.arrow"))
        for file_path in tqdm(files, desc=f"Filtering {split_name}"):
            with pa.memory_map(str(file_path), "r") as source:
                reader = pa.ipc.open_file(source)
                table = reader.read_all()

            df = table.to_pandas()
            daily_max_rv = df.groupby(["Tk", "date"])["rv"].max()
            print(daily_max_rv.describe(
                percentiles=[.01, .05, .1, .25, .5, .75, .9, .95, .99]
            ))
            df = filter_data(df)

            output_path = split_output / file_path.name
            table = pa.Table.from_pandas(
                df,
                preserve_index=False
            )

            with pa.OSFile(str(output_path), "wb") as sink:
                with pa.ipc.new_file(sink, table.schema) as writer:
                    writer.write_table(table)

if __name__ == "__main__":
    preprocess_data(path_data_filler, path_data_preprocessor, DATE_RANGE, SPLIT)
    #print(debug())
    #refilter_arrow_files("preprocessed_data/10p/data_1min_2021_2026", "preprocessed_data/data_1min_2021_2026")
