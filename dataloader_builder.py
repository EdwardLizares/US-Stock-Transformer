import duckdb
import pandas as pd 
from stock_gpt import StockDataset

import torch
from torch.utils.data import DataLoader
from pathlib import Path

from setup import SPLIT, BAR_PER_DAY, INPUT_FEATURES, TARGET_FEATURES, BATCH_SIZE
from setup import path_data_preprocessor
from setup import StockGPT_cfg as C

def test_val_train_splits(con, source_folder: str, split: list[int] = [0.75, 0.90]) -> dict[duckdb.DuckDBPyRelation]:
    """
    Applies split to total dates per ticker and returns a dictionary of dq
    """
    #* Fetches all dates
    dates = con.sql("""
        SELECT DISTINCT date
        FROM full_data
        ORDER BY date
    """).fetchall()
    dates = [x[0] for x in dates]

    train_idx = int(len(dates) * split[0])
    val_idx   = int(len(dates) * split[1])
    train_end = dates[train_idx]
    val_end   = dates[val_idx]

    #* Creates data splits by date
    train = con.sql(f"""
        SELECT *
        FROM full_data
        WHERE date <= DATE '{train_end}'
    """)
    val = con.sql(f"""
        SELECT *
        FROM full_data
        WHERE date > DATE '{train_end}'
            AND date < DATE '{val_end}'
    """)
    test = con.sql(f"""
        SELECT *
        FROM full_data
        WHERE date >= DATE '{val_end}'
    """)

    return {"train": train, "val": val, "test": test}

def calculate_training_norms(con, train: duckdb.DuckDBPyRelation):
    avg_input_cols = ", ".join(f"AVG({col}) AS {col}" for col in INPUT_FEATURES)
    std_input_cols = ", ".join("STDDEV_SAMP({col}) AS {col}" for col in INPUT_FEATURES)
    avg_target_cols = ", ".join(f"AVG({col}) AS {col}" for col in TARGET_FEATURES)
    std_target_cols = ", ".join("STDDEV_SAMP({col}) AS {col}" for col in TARGET_FEATURES)

    #* Average for input columns replacing "f" and "bf" to 0
    avg_input = con.sql(f"""
        SELECT {avg_input_cols} REPLACE (
            0 AS f
            0 AS fb
        )
        FROM train
    """)
    #* STD for input columns replacing "f" and "bf" to 1
    std_input = con.sql(f"""
        SELECT {std_input_cols} REPLACE (
            0 AS f
            0 AS fb
        )
        FROM train
    """)
    #* Average for target columns
    avg_target = con.sql(f"""
        SELECT {avg_target_cols} 
        FROM train
    """)
    #* STD for target columns
    std_target = con.sql(f"""
        SELECT {std_target_cols}
        FROM train
    """)
    return [torch.tensor(x.df().to_numpy(), dtype=torch.float32)
        for x in (avg_input, std_input, avg_target, std_target)
    ]

def build_dataloaders(source_folder: str, split: str = SPLIT,
                      batch_size = BATCH_SIZE, shuffle = True, drop_last = True,
                      num_workers = 2, pin_memory = True, persistent_workers = True):
    print(f"Creating VIEW object from parquets in {source_folder}...")
    #* full_data contains all batches combined as a view
    con = duckdb.execute(f"""
    CREATE VIEW full_data AS
    SELECT *
    FROM read_parquet(
        '{source_folder}/*.parquet'
    )
    """)

    data_queries = test_val_train_splits(con, source_folder)

    train_norms = calculate_training_norms(con, data_queries["train"])

    datasets = [StockDataset(con, dq) for dq in data_queries]

    print(f"Building DataLoaders...")
    train_dl, val_dl, test_dl = [DataLoader(ds, batch_size = batch_size, shuffle = shuffle, drop_last = drop_last,
                                            num_workers = num_workers, pin_memory = pin_memory, persistent_workers=persistent_workers
                                            ) for ds in datasets] #! HARD CODED STUFF

    return train_dl, val_dl, test_dl, train_norms

if __name__ == "__main__":
    train_dl, val_dl, test_dl, train_norms = build_dataloaders(path_data_preprocessor)
    trdli = iter(train_dl)
    tr_inputs, tr_targets = next(trdli)
    print(tr_inputs)