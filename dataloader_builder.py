import pandas as pd 
from stock_gpt import StockDataset

from torch.utils.data import DataLoader
import torch

from setup import SPLIT, BAR_PER_DAY
from setup import path_data_preprocessor

def test_val_train_splits(df: pd.DataFrame, bpd: int, split: list[int]) -> list[pd.DataFrame]:
    """
    Applies split to total dates per ticker
    Handles indivisibility by reducing validation size
    """
    n = len(df)//bpd
    train_rbound = (int(n*split[0]/100)+1)*bpd
    test_lbound = (int(n*split[2]/100)-1)*bpd
    return [df[:train_rbound], df[train_rbound:test_lbound], df[test_lbound:]]

def build_datasets(source_path: str, bpd: int, split: str) -> list[StockDataset]:
    df = pd.read_parquet(source_path)
    train_df, val_df, test_df = test_val_train_splits(df, bpd, split)

    train_ds = StockDataset(train_df)
    val_ds = StockDataset(val_df)
    test_ds = StockDataset(test_df)

    return train_ds, val_ds, test_ds

def build_dataloader(ds: StockDataset):
    return DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0
    ) #! CONTAINS HARDCODED VALUES

def build_dataloaders(source_path: str, bpd: int, split: str) -> list[DataLoader]:
    train_ds, val_ds, test_ds = build_datasets(source_path, bpd, split)

    train_dl = build_dataloader(train_ds)
    val_dl = build_dataloader(val_ds)
    test_dl = build_dataloader(test_ds)

    return train_dl, val_dl, test_dl

if __name__ == "__main__":
    train_dl, val_dl, test_dl = build_dataloaders(path_data_preprocessor, BAR_PER_DAY, SPLIT)
    trdli = iter(train_dl)
    tr_inputs, tr_targets = next(trdli)
    print(trdli)