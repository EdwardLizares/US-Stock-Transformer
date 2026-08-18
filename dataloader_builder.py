import duckdb
import pandas as pd 

import torch
from torch.utils.data import DataLoader

from setup import SPLIT, INPUT_FEATURES, TARGET_FEATURES, BATCH_SIZE, NUM_WORKERS, PERSISTENT_WORKERS
from setup import path_data_preprocessor
from setup import StockGPT_cfg as C
from stock_dataset import StockDataset

def train_val_test_split(source_folder, split: list[int] = [0.75, 0.90]) -> dict[str]:
    """
    Returns WHERE conditionals to append to connection calls for train, val, and test as a dictionary
    """
    #* Fetches all dates
    dates = duckdb.sql(f"""
        SELECT DISTINCT date
        FROM read_parquet('{source_folder}/*.parquet')
        ORDER BY date
    """).fetchall()

    dates = [x[0] for x in dates]
    train_idx = int(len(dates) * split[0])
    val_idx   = int(len(dates) * split[1])
    train_end = dates[train_idx]
    val_end   = dates[val_idx]

    return {"train": f"date <= DATE '{train_end}'",
            "val": f"date < DATE '{val_end}' AND date > DATE '{train_end}'",
            "test": f"date >= DATE '{val_end}'"}

def calculate_training_norms(source_folder, train_cond: str):
    stats_cols = []
    #* SELECT TO CALCULATE/SET INPUT AVG
    for col in INPUT_FEATURES:
        if col in ["f", "fb"]:
            stats_cols.append(f"0 AS input_mean_{col}")
        else:
            stats_cols.append(f"AVG({col}) AS input_mean_{col}")
    #* SELECT TO CALCULATE/SET INPUT STD
    for col in INPUT_FEATURES:
        if col in ["f", "fb"]:
            stats_cols.append(f"1 AS input_std_{col}")
        else:
            stats_cols.append(f"STDDEV_SAMP({col}) AS input_std_{col}")
    #* SELECT TO CALCULATE/SET TARGET AVG
    for col in TARGET_FEATURES:
        stats_cols.append(f"AVG({col}) AS target_mean_{col}")
    #* SELECT TO CALCULATE/SET INPUT STD
    for col in TARGET_FEATURES:
        stats_cols.append(f"STDDEV_SAMP({col}) AS target_std_{col}")
    stats_cols = ", ".join(stats_cols)

    con = duckdb.connect()
    stats = con.execute(f"""
        SELECT {stats_cols}
        FROM read_parquet('{source_folder}/*.parquet')
        WHERE {train_cond}
    """).fetchone()
    con.close()

    ni = len(INPUT_FEATURES)
    nt = len(TARGET_FEATURES)

    input_mean = stats[:ni]
    input_std = stats[ni:2*ni]
    target_mean = stats[2*ni:2*ni + nt]
    target_std = stats[2*ni + nt:]
    #print(input_mean, input_std, target_mean, target_std)
    return [torch.tensor(x, dtype=torch.float32)
            for x in (input_mean, input_std, target_mean, target_std)]

def build_dataloaders(source_folder: str, split: str = SPLIT,
                      batch_size = BATCH_SIZE, drop_last = True,
                      num_workers = NUM_WORKERS, pin_memory = True, persistent_workers = PERSISTENT_WORKERS):
    print(f"Creating VIEW object from parquets in {source_folder}...")
    split_cond = train_val_test_split(source_folder, split)
    train_norms = calculate_training_norms(source_folder, split_cond["train"])
    datasets = [StockDataset(source_folder, split_cond[key]) for key in split_cond.keys()]

    print(f"Building DataLoaders...")
    train_dl, val_dl, test_dl = [DataLoader(ds, batch_size = batch_size, drop_last = drop_last, num_workers = num_workers,
                                            pin_memory = pin_memory, persistent_workers=persistent_workers
                                            ) for ds in datasets] #! HARD CODED STUFF

    return train_dl, val_dl, test_dl, train_norms

if __name__ == "__main__":
    train_dl, val_dl, test_dl, train_norms = build_dataloaders(path_data_preprocessor)
    trdli = iter(train_dl)
    tr_inputs, tr_targets = next(trdli)
    print(tr_inputs)