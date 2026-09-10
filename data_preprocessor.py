import os
from posixpath import split
import pyarrow as pa
import pandas as pd
import numpy as np

from tqdm import tqdm
from pathlib import Path  
from concurrent.futures import ProcessPoolExecutor, as_completed

from setup import AVG_VOLUME_PERIOD, RV_THRESH, MN, MX, RTH_BARS, PM_BARS, INPUT_FEATURES, DATE_RANGE, SPLIT, DEBUG
from setup import path_data_filler, path_data_preprocessor

class ProcessingError(Exception):
    pass

def calculate_additional_hyperparameters(df: pd.DataFrame, pbar = None) -> pd.DataFrame:
    if pbar is not None:
        pbar.set_description("Doing some feature engineering...".ljust(80))

    tk_groups = df.groupby(["Tk"])

    df["av"] = (
        tk_groups["v"].transform(
            lambda x: x.rolling(AVG_VOLUME_PERIOD).mean()
            )
        ).astype("float32")
    df["ema9"] = (
        tk_groups["c"].transform(
            lambda x: x.ewm(span=9, adjust=False).mean()
        )
    ).astype("float32")
    df["ema20"] = (
        tk_groups["c"].transform(
            lambda x: x.ewm(span=20, adjust=False).mean()
        )
    ).astype("float32")
    df["ema12"] = (
        tk_groups["c"].transform(
            lambda x: x.ewm(span=12, adjust=False).mean()
        )
    ).astype("float32")
    df["ema26"] = (
        tk_groups["c"].transform(
            lambda x: x.ewm(span=26, adjust=False).mean()
        )
    ).astype("float32")
    df["macd"] = (df["ema12"]-df["ema26"]).astype("float32")
    df["rv"] = (df["v"] / df["av"]).astype("float32")
    df.drop(columns=["ema12", "ema26", "av"], inplace=True)

    daily_close = (df.groupby(["Tk", "date"])["c"]
                   .last().rename("daily_close").reset_index())
    daily_close["prev_close"] = (daily_close.groupby("Tk")["daily_close"].shift(1))
    df = df.merge(daily_close[["Tk", "date", "prev_close"]],
                  on=["Tk", "date"], how="left")

    df["gp"] = ((df["c"] / df["prev_close"]) - 1).astype("float32")
    df = df.drop(columns=["prev_close"])

    times = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert("America/New_York")
    df["bar"] = times.dt.hour * 60 + times.dt.minute - (9 * 60 + 30) + 1
    return df

def calculate_ibkr_rv(df: pd.DataFrame, pbar=None) -> pd.DataFrame:
    if pbar is not None:
        pbar.set_description("Calculating IBKR RV...".ljust(80))

    daily_volume = (
        df.groupby(["Tk", "date"])["v"].sum()
        .rename("daily_v").reset_index()
    )
    daily_volume = daily_volume.sort_values(["Tk", "date"])
    daily_volume["avg_daily_v"] = (
        daily_volume.groupby("Tk")["daily_v"].transform(
            lambda x: x.shift(1).rolling(90).mean()
        )
    ).astype("float32")

    df = df.merge(daily_volume[["Tk", "date", "avg_daily_v"]], 
                  on=["Tk", "date"], how="left")
    
    df = df.sort_values(["Tk", "t"])
    df["cum_v"] = (df.groupby(["Tk", "date"])["v"].cumsum())
    df["ibkr_rv"] = (df["cum_v"] / df["avg_daily_v"]).astype("float32")
    df.drop(columns=["cum_v", "avg_daily_v"], inplace=True)
    return df

def calculate_ibkr_rvol(df, pbar = None):
    # 1-minute close-to-close log returns, reset each day
    prev_c = df.groupby(["Tk","date"])["c"].shift(1)
    ret = np.log(df["c"] / prev_c).fillna(0)

    rvol30 = ret.groupby(df["Tk"]).transform(
        lambda x: x.rolling(30, min_periods=30).std()
    )
    
    # One historical volatility observation per ticker-day
    day_rvol30 = rvol30.groupby([df["Tk"],df["date"]]).mean()

    # Historical normal = previous 90 ticker-days only
    avg_rvol30 = day_rvol30.groupby(level=0).transform(
        lambda x: x.shift(1).rolling(90, min_periods=90).mean()
    )

    # Broadcast daily historical baseline back onto minute rows
    baseline = pd.MultiIndex.from_arrays([df["Tk"],df["date"]]).map(avg_rvol30)

    df["ibkr_rvol"] = (rvol30 / baseline).astype("float32")
    return df

def calculate_ibkr_rr(df: pd.DataFrame, pbar=None) -> pd.DataFrame:
    if pbar is not None:
        pbar.set_description("Calculating IBKR RR...".ljust(80))

    daily_range = (
        df.groupby(["Tk", "date"])
        .agg(daily_h=("h", "max"), daily_l=("l", "min"), daily_c=("c", "last"))
        .reset_index()
    )
    daily_range["daily_r"] = (daily_range["daily_h"] - daily_range["daily_l"]) / daily_range["daily_c"]
    daily_range = daily_range.sort_values(["Tk", "date"])
    daily_range["avg_daily_r"] = (
        daily_range.groupby("Tk")["daily_r"].transform(
            lambda x: x.shift(1).rolling(90).mean()
        )
    ).astype("float32")

    df = df.merge(daily_range[["Tk", "date", "avg_daily_r"]],
                  on=["Tk", "date"], how="left")

    df = df.sort_values(["Tk", "t"])
    g = df.groupby(["Tk", "date"])
    df["cum_h"] = g["h"].cummax()
    df["cum_l"] = g["l"].cummin()
    df["cum_r"] = (df["cum_h"] - df["cum_l"]) / df["c"]
    df["ibkr_rr"] = (df["cum_r"] / df["avg_daily_r"]).astype("float32")
    df.drop(columns=["cum_h", "cum_l", "cum_r", "avg_daily_r"], inplace=True)
    return df

def engineer_data(df: pd.DataFrame, pbar = None) -> pd.DataFrame:
    df = calculate_additional_hyperparameters(df, pbar)
    df = calculate_ibkr_rv(df, pbar)
    #counts = df.groupby(["Tk", "date"]).size()
    #print(counts.describe())
    #print(counts.value_counts().head())
    return df

def filter_data(df: pd.DataFrame, pbar=None) -> pd.DataFrame:
    if pbar is not None:
        pbar.set_description("Filtering data...".ljust(80))

    g = df.groupby(["Tk", "date"])

    day_low = g["l"].transform("min")
    day_high = g["h"].transform("max")
    rv_count = g["rv"].transform("count")
    ibkr_rv_count = g["ibkr_rv"].transform("count")
    max_ibkr_rv = g["ibkr_rv"].transform("max")

    base_mask = (
        (((day_high - day_low) / day_low) >= 0.05)
        & (day_low <= MX)
        & (day_high >= MN)
        & (rv_count >= RTH_BARS + PM_BARS)
        & (ibkr_rv_count >= RTH_BARS + PM_BARS)
    )

    before_rv = df[base_mask].groupby(["Tk", "date"]).ngroups
    df = df[base_mask & (max_ibkr_rv >= RV_THRESH)]

    print(f"IBKR_RV Filter: {df.groupby(['Tk', 'date']).ngroups}/{before_rv}")
    return df

def preprocess_dataframe(df, apply_filter = True):
    print(f"Loaded: {df.memory_usage(deep=True).sum() / 1024**3:.2f} GB")
    float_cols = df.select_dtypes(include=["float64"]).columns
    df[float_cols] = df[float_cols].astype("float32")
    print(f"Loaded: {df.memory_usage(deep=True).sum() / 1024**3:.2f} GB")
    df = engineer_data(df, None)
    print(f"Loaded: {df.memory_usage(deep=True).sum() / 1024**3:.2f} GB")

    pm = df[(df["bar"] >= -29) & (df["bar"] <= 0) & (df["f"] == 0)]
    actual_pm_days = pm[["Tk", "date"]].drop_duplicates().shape[0]
    total_days = df[["Tk", "date"]].drop_duplicates().shape[0]
    print(f"\nPremarket Data: {actual_pm_days}/{total_days} ticker-days")

    if apply_filter:
        df = filter_data(df, None)
        print(f"Loaded: {df.memory_usage(deep=True).sum() / 1024**3:.2f} GB")

    df = df.sort_values(["date", "Tk", "bar"])
    df = df[INPUT_FEATURES+["Tk", "date"]]

    float_cols = df.select_dtypes(include=["float64"]).columns
    df[float_cols] = df[float_cols].astype("float32")
    return df


def preprocess_file(file_path, output_folder, split, split_names, train_end = None, val_end = None):
    output_paths = {split_name: output_folder/ split_name / file_path.with_suffix(".arrow").name
                    for split_name in split_names}
    if all(path.exists() for path in output_paths.values()):
        return

    df = pd.read_parquet(file_path)
    df = preprocess_dataframe(df)
    
    print(df[['v', 'ibkr_rv', 'rv', 'Tk']])
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
            df = filter_data(df)

            output_path = split_output / file_path.name
            table = pa.Table.from_pandasa(
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
