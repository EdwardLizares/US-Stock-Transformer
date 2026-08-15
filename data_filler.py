import os
import pandas as pd 
import duckdb as duckdb
from tqdm import tqdm

from setup import DATE_RANGE, BAR_WIDTH
from setup import path_data_scrapper, path_data_filler

def get_unix_timestamps(date_range: pd.DatetimeIndex, bar_width: int):
    return pd.to_datetime([
        f"{date} {time}"
        for date in date_range.strftime("%Y-%m-%d")
        for time in pd.date_range(
                        f"09:{30+bar_width}",
                        f"16:00",
                        freq=f"{bar_width}min"
                    ).strftime("%H:%M")
        ], format="%Y-%m-%d %H:%M").tz_localize("America/New_York").as_unit("ms").astype("int64")

#! Switching to duckdb
def chunk_parquet(dfs, chunk_id, output_root, limit=3000):
    os.makedirs(f"{output_root}", exist_ok=True)
    if len(dfs) >= limit:
        chunk_df = pd.concat(dfs, ignore_index=True)
        chunk_df.to_parquet(
            f"{output_root}/chunk{chunk_id}.parquet",
            index=False
        )
        dfs.clear()
        chunk_id += 1
    return chunk_id

def forward_back_fill(source_path: str):
    duckdb.sql("""
        SELECT *
        FROM 'source_path.parquet'
         
    """)
     
def old_forward_back_fill(df: pd.DataFrame, date_range: pd.DatetimeIndex,
                      bar_width: int, output_root: str) -> pd.DataFrame:
    chunk_id, dfs = 0, []
    tkrs = df.index.get_level_values("T").unique()
    misses, hits, back_fills, n = 0, 0, 0, len(tkrs)
    pbar = tqdm(tkrs, total=n, desc=f"Forward filling...".ljust(80),
                bar_format="|{bar}| {percentage:3.1f}% ({elapsed}) {desc}")
    for tkr in tkrs: #! Forward fill & backfill per ticker per day
        tkr_df = df.loc[tkr]
        expected_timestamps = get_unix_timestamps(date_range, bar_width)   
        ety = tkr_df.reindex(expected_timestamps)
        ety["date"] = (pd.to_datetime(ety.index, unit="ms", utc=True).tz_convert("America/New_York").date)
        ety["f"] = 0                #! FAKE/SYTHETIC BAR    
        ety["bf"] = 0               #! BACKFILLED SPECIFICALLY (HOT ENCODING)

        f_mask = ety["c"].isna()
        misses += f_mask.sum()
        hits += (~f_mask).sum()
        p_c = ety.groupby("date")["c"].ffill()
        for col in ["o", "h", "l", "c", "vw"]:
            ety.loc[f_mask, col] = p_c[f_mask]
        ety.loc[f_mask, ["v", "n"]] = 0
        ety.loc[f_mask, ["f"]] = 1

        b_mask = p_c.isna()
        back_fills += b_mask.sum()
        if b_mask.any():
            n_o = ety.groupby("date")["o"].transform("first")
            for col in ["o", "h", "l", "c", "vw"]:
                        ety.loc[b_mask, col] = n_o[b_mask]
            ety.loc[b_mask, ["v", "n"]] = 0
            ety.loc[b_mask, ["bf"]] = 1

        ety["T"] = tkr
        ety["t"] = ety.index

        dfs.append(ety)
        chunk_id = chunk_parquet(dfs, chunk_id, output_root)

        pbar.update(1)
        pbar.set_description_str((f"[{pbar.n}/{n}] "
                                  f"{{{misses}|{hits}|{back_fills}}} "
                                  f"Finished filling data for {tkr}".ljust(80)))
    chunk_parquet(dfs, chunk_id, output_root, 0)
    return pd.read_parquet(f"{output_root}/chunk0.parquet")

def old_fill_raw_data(source_path: str, output_root: str) -> pd.DataFrame:
    """
    Forward/back filled data for all possible timestamps
    Creates a .parquet of the DataFrame on first run
    """
    try:
        return pd.read_parquet(f"{output_root}/chunk0.parquet")
    except FileNotFoundError as _:
        pass
    
    print("Reading source path parquet...")
    df = pd.read_parquet(source_path)

    df = df.set_index(["T", "t"])
    df = forward_back_fill(df, DATE_RANGE, BAR_WIDTH, output_root)


if __name__ == "__main__":
    filled_raw_data = fill_raw_data(path_data_scrapper, path_data_filler)
    print(filled_raw_data)
