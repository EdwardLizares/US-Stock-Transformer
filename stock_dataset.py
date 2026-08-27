import torch
import pyarrow as pa
import numpy as np

from torch.utils.data import Dataset, get_worker_info
from pathlib import Path

from setup import StockBPT_cfg as cfg

class StockDataset(Dataset):
    def __init__(self, source_folder, file_limit = cfg["file_limit"], seq_len: int = cfg["seq_len"],
                 step: int = cfg["step"], bar_per_day = cfg["bar_per_day"],
                 input_features: list = cfg["input_features"], target_features: list = cfg["target_features"]):
        """
        Converts a folder of arrow files into a dataset
        """
        super().__init__()

        self.source_folder = Path(source_folder)

        self.input_features = input_features
        self.target_features = target_features
        self.target_indices = [
            self.input_features.index(feature)
            for feature in self.target_features
        ]

        self.seq_len = seq_len
        self.step = step
        self.bar_per_day = bar_per_day

        #* Number of windows produced by one ticker-day
        self.samples_per_day = (bar_per_day - seq_len - step + 1)

        self.files = sorted(self.source_folder.glob("*.arrow"))[:file_limit]

        self.samples_per_file = []
        for file_path in self.files:
            with pa.memory_map(str(file_path), "r") as source:
                reader = pa.ipc.open_file(source)
                n_rows = sum(
                    reader.get_batch(i).num_rows for i in range(reader.num_record_batches)
                )

            n_days = n_rows // self.bar_per_day
            self.samples_per_file.append(n_days * self.samples_per_day)

        self.offsets = np.cumsum([0] + self.samples_per_file)
        self.cached_file_idx = None
        self.cached_source = None
        self.cached_table = None

    def __len__(self):
        return int(self.offsets[-1])

    def _load_file(self, file_idx):
        """
        Memory maps an Arrow shard. If this shard is already open, do nothing.
        """
        if file_idx == self.cached_file_idx:
            return
        if self.cached_source is not None:
            self.cached_source.close()
        self.cached_source = pa.memory_map(str(self.files[file_idx]),"r")

        reader = pa.ipc.open_file(self.cached_source)
        self.cached_table = reader.read_all()
        self.cached_file_idx = file_idx

    def __getitem__(self, idx):
        file_idx = np.searchsorted(self.offsets, idx, side="right") - 1
        local_idx = idx - self.offsets[file_idx]
        day_idx = local_idx // self.samples_per_day
        window_idx = local_idx % self.samples_per_day

        row_start = day_idx * self.bar_per_day + window_idx
        self._load_file(file_idx)

        x_table = self.cached_table.slice(row_start, self.seq_len)
        y_table = self.cached_table.slice(row_start + self.step, self.seq_len)

        x = np.column_stack([
            x_table[col].to_numpy() for col in self.input_features
        ]).astype(np.float32, copy=False)

        y = np.column_stack([
            y_table[col].to_numpy() for col in self.target_features
        ]).astype(np.float32, copy=False)
        y = y - x[:, self.target_indices]

        return torch.from_numpy(x), torch.from_numpy(y)

    def __del__(self):
        if self.cached_source is not None:
            self.cached_source.close()