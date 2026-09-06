import torch
import numpy as np
import pyarrow as pa

from torch.utils.data import DataLoader
from pathlib import Path

from setup import INPUT_FEATURES, BATCH_SIZE, NUM_WORKERS, PERSISTENT_WORKERS, FILE_LIMIT
from setup import path_data_preprocessor
from stock_dataset_mpt import StockDatasetMPT

def calculate_training_norms(source_folder, input_features):
    """
    Expects source folder to be all train arrow files and returns all calculated norms
    """
    #* Accumulates across all train files
    input_sum = np.zeros(len(input_features), dtype=np.float64)
    input_sq_sum = np.zeros(len(input_features), dtype=np.float64)

    n = 0
    files = sorted(Path(source_folder).glob("*.arrow"))

    if FILE_LIMIT is not None:
        files = files[:FILE_LIMIT]
    for file_path in files:
        with pa.memory_map(str(file_path), "r") as source:
            table = pa.ipc.open_file(source).read_all()
            x = np.column_stack(
                [table[col].to_numpy() for col in input_features]
            ).astype(np.float64, copy=False)

            input_sum += x.sum(axis=0)
            input_sq_sum += (x ** 2).sum(axis=0)
            n += len(x)

    input_mean = input_sum / n
    input_std = np.sqrt(input_sq_sum / n - input_mean**2)

    if "f" in input_features:
        i = input_features.index("f")
        input_mean[i] = 0.0
        input_std[i] = 1.0

    return [
        torch.tensor(input_mean, dtype=torch.float32),
        torch.tensor(input_std, dtype=torch.float32),
    ]

def build_dataloaders(source_folder: str, train_val = True,
                      input_features = INPUT_FEATURES,
                      batch_size = BATCH_SIZE,
                      drop_last = True,
                      num_workers = NUM_WORKERS,
                      pin_memory = True,
                      persistent_workers = PERSISTENT_WORKERS):
    """
    Returns a dictionary of train-val-test dataloaders and the training norms
    """
    train_norms = None
    if train_val:
        train_norms = calculate_training_norms(f"{source_folder}/train", input_features)

    datasets = {subfolder.name : StockDatasetMPT(subfolder) for subfolder in list(Path(source_folder).glob("*/"))}

    print(f"Building DataLoaders...")
    dataloaders = {ds_key : DataLoader(datasets[ds_key], batch_size = batch_size, drop_last = drop_last, 
                                       num_workers = num_workers, persistent_workers=persistent_workers,
                                       pin_memory = pin_memory) for ds_key in datasets.keys()}
    if train_val:
        print(f"Train dataset samples: {len(datasets['train']):,}")
        print(f"Train loader batches:  {len(dataloaders['train']):,}")
        print(f"Batch size:            {batch_size}")
    return dataloaders, train_norms

if __name__ == "__main__":
    dataloaders, train_norms = build_dataloaders(path_data_preprocessor, False)