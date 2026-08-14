import pandas as pd 
from stock_gpt import StockDataset

import torch
from torch.utils.data import DataLoader

from setup import SPLIT, BAR_PER_DAY, INPUT_FEATURES, TARGET_FEATURES, BATCH_SIZE
from setup import path_data_preprocessor
from setup import Stock_GPT_cfg as C

def test_val_train_splits(df: pd.DataFrame, bpd: int, split: list[int]) -> list[pd.DataFrame]:
    """
    Applies split to total dates per ticker
    Handles indivisibility by reducing validation size
    """
    n = len(df)//bpd
    train_rbound = (int(n*split[0])+1)*bpd
    test_lbound = (int(n*split[1])-1)*bpd
    return [df[:train_rbound], df[train_rbound:test_lbound], df[test_lbound:]]

def calculate_training_norms(df: pd.DataFrame):
    i_mu = df[INPUT_FEATURES].mean()
    i_mu[["f", "bf"]] = 0
    i_sig = df[INPUT_FEATURES].std()
    i_sig[["f", "bf"]] = 1
    t_mu = df[TARGET_FEATURES].mean()
    t_sig = df[TARGET_FEATURES].std()

    return [torch.tensor(x.to_numpy(), dtype=torch.float32)
        for x in (i_mu, i_sig, t_mu, t_sig)
    ]

def build_dataloaders(source_path: str, bpd: int = BAR_PER_DAY, 
                      split: str = SPLIT, num_workers = 2):
    print(f"Reading source path at {source_path}...")
    df = pd.read_parquet(source_path)
    dataframes = test_val_train_splits(df, bpd, split)

    train_norms = calculate_training_norms(dataframes[0])

    datasets = [StockDataset(df) for df in dataframes]

    print(f"Building DataLoaders...")
    train_dl, val_dl, test_dl = [DataLoader(ds, batch_size = BATCH_SIZE, shuffle = True, drop_last = True,
                                            num_workers = 2, pin_memory = True, persistent_workers=True
                                            ) for ds in datasets] #! HARD CODED STUFF

    return train_dl, val_dl, test_dl, train_norms

if __name__ == "__main__":
    train_dl, val_dl, test_dl, train_norms = build_dataloaders(path_data_preprocessor)
    trdli = iter(train_dl)
    tr_inputs, tr_targets = next(trdli)
    print(tr_inputs)