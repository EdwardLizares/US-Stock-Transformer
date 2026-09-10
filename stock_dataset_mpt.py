import torch
import pyarrow as pa
import numpy as np

from torch.utils.data import Dataset, get_worker_info
from pathlib import Path

from setup import StockMPT_cfg as cfg

class StockDatasetMPT(Dataset):
    def __init__(self, source_folder, file_limit = cfg["file_limit"], seq_len: int = cfg["seq_len"],
                 step: int = cfg["step"], rth_bars = cfg["rth_bars"], pm_bars = cfg["pm_bars"],
                 input_features: list = cfg["input_features"], target_features: list = cfg["target_features"]):
        """
        Converts a folder of arrow files into a dataset
        """
        super().__init__()

        self.source_folder = Path(source_folder)

        self.input_features = input_features
        self.target_features = target_features

        self.step = step
        self.rth_bars = rth_bars
        self.pm_bars = pm_bars
        self.total_bars = rth_bars + pm_bars
        self.seq_len = self.total_bars - step
        self.samples_per_day = 1

        self.files = sorted(self.source_folder.glob("*.arrow"))[:file_limit]

        self.samples_per_file = []
        for file_path in self.files:
            with pa.memory_map(str(file_path), "r") as source:
                reader = pa.ipc.open_file(source)
                n_rows = sum(
                    reader.get_batch(i).num_rows for i in range(reader.num_record_batches)
                )

            n_days = n_rows // self.total_bars
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

        day_idx = local_idx
        row_start = day_idx * self.total_bars

        self._load_file(file_idx)

        x_table = self.cached_table.slice(row_start, self.seq_len)
        y_table = self.cached_table.slice(row_start + self.step, self.seq_len)

        x = np.column_stack([
            x_table[col].to_numpy() for col in self.input_features
        ]).astype(np.float32, copy=False)

        future = y_table["c"].to_numpy().astype(np.float32, copy=False)
        current = x_table["c"].to_numpy().astype(np.float32, copy=False)

        change = (future - current) # / current

        constant_threshold = 0.01
        percent_threshold = 0.01  # 0.1%

        y = np.ones(self.seq_len, dtype=np.int64)
        y[change < -constant_threshold] = 0
        y[change > constant_threshold] = 2

        y = y[self.pm_bars:]

        return torch.from_numpy(x), torch.from_numpy(y)

    def get_metadata(self, idx):
        file_idx = np.searchsorted(self.offsets, idx, side="right") - 1
        local_idx = idx - self.offsets[file_idx]
        day_idx = local_idx

        row_start = day_idx * self.total_bars
        self._load_file(file_idx)

        table = self.cached_table.slice(row_start, self.total_bars)

        return {
            "Tk": table["Tk"][0].as_py(),
            "date": table["date"][0].as_py()
        }

    def __del__(self):
        if self.cached_source is not None:
            self.cached_source.close()