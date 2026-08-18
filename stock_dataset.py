import torch
import pyaroow

from torch.utils.data import Dataset, get_worker_info
from pathlib import Path

from setup import StockGPT_cfg as cfg

class StockDataset(Dataset):
    def __init__(self, source_folder, split_cond: str, seq_len: int = cfg["seq_len"],
                 step: int = cfg["step"], bar_per_day = cfg["bar_per_day"],
                 input_features: list = cfg["input_features"], target_features: list = cfg["target_features"]):
        """
        Converts a duckdb relation object into a dataset
        """
        super().__init__()
        self.input_features = input_features
        self.target_features = target_features

        self.source_folder = source_folder
        self.split_cond = split_cond

        self.seq_len = seq_len
        self.step = step
        self.bar_per_day = bar_per_day

    def __len__(self):
        return self.length

    def __getitem__(self):
        