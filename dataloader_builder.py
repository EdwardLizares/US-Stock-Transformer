import torch
import numpy as np
import pyarrow as pa

from torch.utils.data import DataLoader
from pathlib import Path

from setup import INPUT_FEATURES, TARGET_FEATURES, BATCH_SIZE, NUM_WORKERS, PERSISTENT_WORKERS, FILE_LIMIT, STEP, BAR_PER_DAY
from setup import path_data_preprocessor
from stock_dataset import StockDataset
def calculate_training_norms(source_folder, input_features, target_features, 
                             file_limit=FILE_LIMIT, step=STEP, bar_per_day=BAR_PER_DAY):
    """
    Expects source folder to be all train arrow files and returns all calculated norms
    """
    #* Accumulates across all train files
    input_sum = np.zeros(len(input_features), dtype=np.float64)
    input_sq_sum = np.zeros(len(input_features), dtype=np.float64)
    target_sum = np.zeros(len(target_features), dtype=np.float64)
    target_sq_sum = np.zeros(len(target_features), dtype=np.float64)

    n = 0
    target_n = 0
    files = sorted(Path(source_folder).glob("*.arrow"))

    if file_limit is not None:
        files = files[:file_limit]
    for file_path in files:
        with pa.memory_map(str(file_path), "r") as source:
            table = pa.ipc.open_file(source).read_all()
            x = np.column_stack(
                [table[col].to_numpy() for col in input_features]
            ).astype(np.float64, copy=False)
            y = np.column_stack(
                [table[col].to_numpy() for col in target_features]
            ).astype(np.float64, copy=False)

            #* Converts to residual %
            n_days = len(y) // bar_per_day
            y = y[:n_days * bar_per_day]
            y = y.reshape(n_days, bar_per_day, len(target_features))
            current = y[:, :-step, :]
            future = y[:, step:, :]
            y = (future - current) / current
            y = y.reshape(-1, len(target_features))

            input_sum += x.sum(axis=0)
            target_sum += y.sum(axis=0)
            input_sq_sum += (x ** 2).sum(axis=0)
            target_sq_sum += (y ** 2).sum(axis=0)
            n += len(x)
            target_n += len(y)

    input_mean = input_sum / n
    target_mean = target_sum / target_n
    input_std = np.sqrt(input_sq_sum / n - input_mean**2)
    target_std = np.sqrt(target_sq_sum / target_n - target_mean**2)

    if "f" in input_features:
        i = input_features.index("f")
        input_mean[i] = 0.0
        input_std[i] = 1.0

    return [
        torch.tensor(input_mean, dtype=torch.float32),
        torch.tensor(input_std, dtype=torch.float32),
        torch.tensor(target_mean, dtype=torch.float32),
        torch.tensor(target_std, dtype=torch.float32),
    ]

def build_dataloaders(source_folder: str, train_val = True,
                      input_features = INPUT_FEATURES,
                      target_features = TARGET_FEATURES,
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
        train_norms = calculate_training_norms(f"{source_folder}/train", input_features, target_features)

    datasets = {subfolder.name : StockDataset(subfolder) for subfolder in list(Path(source_folder).glob("*/"))}

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