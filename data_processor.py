from data_scrapper_per_ticker import data_scrapper
import pandas as pd 
import pandas_market_calendars as mcal
from tqdm import tqdm
from itertools import product

# Cameron-Lizares Screener -> Volume > 5xAvg Volume, Last$ = 1-20, Gap% > 0
FILE_NAME = "data_15min_2025"
AVG_VOLUME_PERIOD = 90
RATIO_THRESHOLD = 5 #! testing, set to 5
HYPERPARAMETER_COMBOS = [["T", "t", "vw", "ema9", "ema20", "o", "c", "h", "l", "n", "rv", "gp", "otc", "y"]]
DATE_RANGE = mcal.get_calendar("NYSE").schedule("2025-01-01","2025-12-31").index
BAR_WIDTH = 15 #! MINUTES

def get_unix_timestamps(date: str, bar_width):
    return pd.to_datetime([
        f"{date} {time}" for time in pd.date_range(
                f"09:30",
                f"16:00",
                freq=f"{bar_width}min"
            ).strftime("%H:%M")
        ], format="%Y-%m-%d %H:%M").tz_localize("America/New_York").as_unit("ms").astype("int64")

def process_data(path_name: str) -> pd.DataFrame:
    df = pd.read_parquet(path_name)
    df = df.set_index(["T", "t"])
    df = forward_fill(df, DATE_RANGE, BAR_WIDTH)
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
    return df[HYPERPARAMETER_COMBOS[id]]

def forward_fill(df: pd.DataFrame, date_range: pd.DatetimeIndex, bar_width: int) -> pd.DataFrame:
    filled_dfs = []
    tkrs, dates = df.index.get_level_values("T").unique(), date_range.strftime('%Y-%m-%d')
    misses, hits, back_fills, n = 0, 0, 0, len(tkrs)
    pbar = tqdm(tkrs, total=n, desc=f"Forward filling...".ljust(80),
                bar_format="|{bar}| {percentage:3.1f}% ({elapsed}) {desc}")
    for tkr in tkrs: #! Forward fill & backfill per ticker per day
        tkr_df = df.loc[tkr]
        for date in dates:
            expected_timestamps = get_unix_timestamps(date, bar_width)   
            et = tkr_df.reindex(expected_timestamps)
            if et["c"].isna().all(): #! FULL DAY EMPTY
                continue

            f_mask = et["c"].isna()
            misses += f_mask.sum()
            hits += (~f_mask).sum()

            p_c = et["c"].ffill()
            for col in ["o", "h", "l", "c", "vw"]:
                et.loc[f_mask, col] = p_c[f_mask]
            et.loc[f_mask, ["v", "n"]] = 0

            b_mask = p_c.isna()
            back_fills += b_mask.sum()
            if b_mask.any():
                n_o = et["o"].dropna().iloc[0]
                for col in ["o", "h", "l", "c", "vw"]:
                            et.loc[b_mask, col] = n_o
                et.loc[b_mask, ["v", "n"]] = 0

            et["T"] = tkr
            et["t"] = et.index
            filled_dfs.append(et)
        pbar.update(1)
        pbar.set_description_str((f"[{pbar.n}/{n}] "
                                  f"{{{misses}|{hits}|{back_fills}}} "
                                  f"Finished filling data for {tkr}".ljust(80)))
    return pd.concat(filled_dfs, ignore_index=True)

def forward_fill_unoptimal(df: pd.DataFrame, date_range: pd.DatetimeIndex, bar_width: int) -> pd.DataFrame:
    df_full = pd.DataFrame()
    tkrs, dates = df.index.get_level_values("T").unique(), date_range.strftime('%Y-%m-%d')
    tkr_dates = product(tkrs, dates)
    misses, hits, back_fills, n = 0, 0, 0, len(tkrs)*len(dates)
    pbar = tqdm(tkr_dates, total=n, desc=f"Forward filling...".ljust(80),
                bar_format="|{bar}| {percentage:3.1f}% ({elapsed}) {desc}")
    for tkr, date in pbar:
        #! Forward fill per day!
        p_c = -1
        idx, timestamps = 0, get_unix_timestamps(date, bar_width)
        pbar.set_description_str((f"[{pbar.n}/{n}] "
                                  f"{{{misses}|{hits}|{back_fills}}} "
                                  f"Checking {tkr} data on {date}".ljust(80)))
        while idx < len(timestamps):
            key = (tkr, timestamps[idx])
            if key in df.index:
                hits += 1
                cur = df.loc[key]
                p_c = cur["c"]
                df_full = pd.concat([df_full, cur.to_frame().T], ignore_index=True)
            else:
                misses += 1
                if p_c != -1: #* Forward-fillable
                    df_full = pd.concat([df_full, pd.DataFrame([{
                        "T": tkr, "t": timestamps[idx], "o": p_c, "c": p_c, "v": 0,
                        "h": p_c, "l": p_c, "n": 0, "otc": None, "vw": p_c #! CONSIDER CHANGING?
                    }])])
                else: #* Back-fillable
                    steps_ahead = 1
                    while idx + steps_ahead < len(timestamps):
                        next_key = (tkr, timestamps[idx+steps_ahead])
                        if next_key in df.index:
                            cur = df.loc[next_key]
                            break
                        steps_ahead += 1
                    else: #! FULL DAY WAS EMPTY
                        break
                    back_fills += steps_ahead
                    n_o = cur["o"]
                    for s in range(steps_ahead):
                        df_full = pd.concat([df_full, pd.DataFrame([{
                                                    "T": tkr, "t": timestamps[idx+s],
                                                    "o": n_o, "c": n_o, "v": 0,
                                                    "h": n_o, "l": n_o, "n": 0, "otc": None, "vw": n_o #! CONSIDER CHANGING?
                                            }])])
                    idx += steps_ahead - 1
            idx += 1
    print(f"Misses: {misses} | Hits {hits} | Back Fills {back_fills}")
    return df_full
            
if __name__ == "__main__":
    #timestamps = get_unix_timestamps("2025-01-02", 15)
    #print(timestamps[0])
    #df = pd.read_parquet(f"raw_data/{FILE_NAME}.parquet")
    #df = df.set_index(["T", "t"])
    #print(df.index.get_level_values("t")[0])
    #print(get_unix_timestamps("2026-01-01", 15))
    processed_data = process_data(f"raw_data/{FILE_NAME}.parquet")
    print(processed_data)
    processed_data.to_parquet(f"processed_data/{FILE_NAME}.parquet", index=False)