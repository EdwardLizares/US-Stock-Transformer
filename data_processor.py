from data_scrapper import data_scrapper
import pandas as pd 
import pandas_market_calendars as mcal
from tqdm import tqdm
from itertools import product

from setup import API_KEY, DATE_RANGE, BAR_WIDTH, HYPERPARAMETERS, RATIO_THRESHOLD
from setup import path_data_scrapper, path_processed_data

def get_unix_timestamps(date_range: pd.DatetimeIndex, bar_width):
    return pd.to_datetime([
        f"{date} {time}"
        for date in date_range.strftime("%Y-%m-%d")
        for time in pd.date_range(
                        f"09:30",
                        f"16:00",
                        freq=f"{bar_width}min"
                    ).strftime("%H:%M")
        ], format="%Y-%m-%d %H:%M").tz_localize("America/New_York").as_unit("ms").astype("int64")

def process_data(path_name: str) -> pd.DataFrame:
    df = pd.read_parquet(path_name)
    df = df.set_index(["T", "t"])
    df = forward_back_fill(df, DATE_RANGE, BAR_WIDTH)
    df = calculate_additional_hyperparameters(df)
    #df = cameron_lizares_screener(df) #! remove for now
    df = select_hyperparameters(df, id=0)
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

def cameron_lizares_screener(df: pd.DataFrame) -> pd.DataFrame:
    #* Filter by ratio, close, gap% ...
    return df[(df["v"]>df["av"]*RATIO_THRESHOLD) &
            (df["Close"].between(1, 20)) &
            (df["Gap%"]>0)]

def select_hyperparameters(df: pd.DataFrame, id: int) -> pd.DataFrame:
    return df[HYPERPARAMETERS]

def forward_back_fill(df: pd.DataFrame, date_range: pd.DatetimeIndex, bar_width: int) -> pd.DataFrame:
    filled_dfs = []
    tkrs = df.index.get_level_values("T").unique()
    misses, hits, back_fills, n = 0, 0, 0, len(tkrs)
    pbar = tqdm(tkrs, total=n, desc=f"Forward filling...".ljust(80),
                bar_format="|{bar}| {percentage:3.1f}% ({elapsed}) {desc}")
    for tkr in tkrs: #! Forward fill & backfill per ticker per day
        tkr_df = df.loc[tkr]
        expected_timestamps = get_unix_timestamps(date_range, bar_width)   
        ety = tkr_df.reindex(expected_timestamps)
        ety["date"] = (pd.to_datetime(ety.index, unit="ms", utc=True).tz_convert("America/New_York").date)
        ety["fb"] = 0

        f_mask = ety["c"].isna()
        misses += f_mask.sum()
        hits += (~f_mask).sum()
        p_c = ety.groupby("date")["c"].ffill()
        for col in ["o", "h", "l", "c", "vw"]:
            ety.loc[f_mask, col] = p_c[f_mask]
        ety.loc[f_mask, ["v", "n"]] = 0
        ety.loc[f_mask, ["fb"]] = 1

        b_mask = p_c.isna()
        back_fills += b_mask.sum()
        if b_mask.any():
            n_o = ety.groupby("date")["o"].transform("first")
            for col in ["o", "h", "l", "c", "vw"]:
                        ety.loc[b_mask, col] = n_o[b_mask]
            ety.loc[b_mask, ["v", "n"]] = 0
            ety.loc[b_mask, ["fb"]] = 2

        ety["T"] = tkr
        ety["t"] = ety.index
        filled_dfs.append(ety)
        pbar.update(1)
        pbar.set_description_str((f"[{pbar.n}/{n}] "
                                  f"{{{misses}|{hits}|{back_fills}}} "
                                  f"Finished filling data for {tkr}".ljust(80)))
    return pd.concat(filled_dfs, ignore_index=True)
   
if __name__ == "__main__":
    processed_data = process_data(path_data_scrapper)
    print(processed_data)
    processed_data.to_parquet(path_data_processor, index=False)