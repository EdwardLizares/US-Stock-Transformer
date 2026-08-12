import pandas as pd

from setup import HYPERPARAMETERS, AVG_VOLUME_PERIOD, RV_THRESH, MN, MX
from setup import path_data_filler, path_data_preprocessor

class ProcessingError(Exception):
    pass

def trim_data(df: pd.DataFrame):
    return df

def calculate_additional_hyperparameters(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["T", "t"])
    df["av"] = (
        df.groupby("T")["v"].transform(
            lambda x: x.rolling(AVG_VOLUME_PERIOD).mean()
            )
        )
    df["ema9"] = (
        df.groupby("T")["c"].transform(
            lambda x: x.ewm(span=9, adjust=False).mean()
        )
    )
    df["ema20"] = (
        df.groupby("T")["c"].transform(
            lambda x: x.ewm(span=20, adjust=False).mean()
        )
    )
    df["ema12"] = (
        df.groupby("T")["c"].transform(
            lambda x: x.ewm(span=12, adjust=False).mean()
        )
    )
    df["ema26"] = (
        df.groupby("T")["c"].transform(
            lambda x: x.ewm(span=26, adjust=False).mean()
        )
    )
    df["macd"] = (df["ema12"]-df["ema26"])
    p_c = df.groupby("T")["c"].shift(1)
    df["gp"] = (df["o"] - p_c) / p_c * 100
    df["cp"] = (df["o"] - df["c"]) / df["c"] * 100
    df["rv"] = df["v"] / df["av"]
    n_o = df.groupby("T")["o"].shift(-1)
    df["y"] = (n_o > df["c"]).astype(int)

    df["bar"] = df.groupby(["T", "date"]).cumcount() + 1
    df.insert(df.columns.get_loc("date") + 1, "bar", df.pop("bar"))

    return df

def select_hyperparameters(df: pd.DataFrame) -> pd.DataFrame:
    return df[HYPERPARAMETERS]

def clear_data(df: pd.DataFrame) -> pd.DataFrame:
    print("Culling empty days...")
    df = df[(df["o"].notna())] #* Should be removing whole days if data processed correctly
    counts = df.groupby(["T", "date"]).size().value_counts()
    if len(counts.values)>1:
        raise ProcessingError("Something went wrong with processing the data")
    return df

def engineer_data(df: pd.DataFrame) -> pd.DataFrame:
    print("Doing some feature engineering...")
    df = calculate_additional_hyperparameters(df)
    df = select_hyperparameters(df)
    return df

def filter_data(df: pd.DataFrame) -> pd.DataFrame: 
    """
    Filters for RV and Price
    Additionally removes days with any NaN rv
    """
    print("Filtering data...")
    rv_mask = (df.groupby(["T", "date"])["rv"].transform("max")>RV_THRESH)
    df = df[rv_mask]
    pc_mask = ((df.groupby(["T", "date"])["l"].transform("min")<=MX) &
               (df.groupby(["T", "date"])["h"].transform("min")>MN))
    df = df[pc_mask]

    nan_mask = (df.groupby(["T", "date"])["rv"].transform("count") < 26)
    df = df[~nan_mask]

    return df

def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    return df

def preprocess_data(source_path: str, output_path: str):
    """
    Returns a DataFrame with preprocessed data
    Creates a .parquet of the DataFrame on first run
    """
    try:
        return pd.read_parquet(output_path)
    except FileNotFoundError as _:
        pass

    print(f"Reading source path parquet at [{source_path}]")
    df = pd.read_parquet(source_path)

    df = clear_data(df)
    df = engineer_data(df)
    df = filter_data(df)
    df = df.set_index(["T", "date"])
    df = normalize_data(df)

    print("Saving Data...")
    df.to_parquet(output_path, index=True)
    return df

def debug(df: pd.DataFrame):
    bc = df["bar"].value_counts().sort_index()
    print(f"Bar Counts: " + 
          ("Valid " if bc.iloc[0]*len(bc)==len(df) else "Invalid") + 
          f"({len(bc)}/{bc.iloc[0]}/{len(df)})")
    print(f"Bar Counts: " + 
          ("Valid " if df.isna().sum().sum()==0 else "Invalid") + 
          f"({df.isna().sum().sum()})")

if __name__ == "__main__":
    preprocessed_data = preprocess_data(path_data_filler, path_data_preprocessor)
    print(preprocessed_data)
    debug(preprocessed_data)
