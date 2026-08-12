import pandas as pd

from setup import SEQ_LEN, AVG_VOLUME_PERIOD, HYPERPARAMETERS
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
    df["gp"] = (df["o"] - df["c"].shift(1)) / df["c"].shift(1) * 100
    df["cp"] = (df["o"] - df["c"]) / df["c"] * 100
    df["rv"] = df["v"] / df["av"]
    df["y"] = (df["o"].shift(-1) > df["c"]).astype(int)
    #! dont remove anything yet
    return df

def select_hyperparameters(df: pd.DataFrame) -> pd.DataFrame:
    return df[HYPERPARAMETERS]

def data_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    print("Culling empty days...")
    df = df[(df["o"].notna())] #* Should be removing whole days if data processed correctly
    counts = df.groupby(["T", "date"]).size().value_counts()
    if (len(counts.values)>1):
        raise ProcessingError("Something went wrong with processing the data")
    print("Doing some feature engineering...")
    df = calculate_additional_hyperparameters(df)
    df = select_hyperparameters(df)
    return df

def preprocess_data(path):
    print("Reading source path parquet...")
    df = pd.read_parquet(path)
    df = data_cleanup(df)
    df.to_parquet(path_data_preprocessor, index=False)
    return df

if __name__ == "__main__":
    dataset = preprocess_data(path_data_filler)
    print(dataset)