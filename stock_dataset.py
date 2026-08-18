import torch
import duckdb

from torch.utils.data import IterableDataset, get_worker_info
from pathlib import Path

from setup import StockGPT_cfg as cfg

class StockDataset(IterableDataset):
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

        con = duckdb.connect()
        n_days = con.execute(f"""
            SELECT COUNT(*)
            FROM (
                SELECT DISTINCT T_1, date
                FROM read_parquet('{source_folder}/*.parquet')
                WHERE {split_cond}
            )
        """).fetchone()[0]
        con.close()
        samples_per_day = (bar_per_day - seq_len - step + 1)
        self.length = n_days * samples_per_day

    def __len__(self):
        return self.length

    def __iter__(self):
        files = list(Path(self.source_folder).glob("*.parquet"))
        worker = get_worker_info()
        if worker is None:
            worker_files = files
        else:
            worker_files = files[worker.id::worker.num_workers]
        con = duckdb.connect()
        for file_path in worker_files:
            df = con.execute(f"""
                SELECT *
                FROM read_parquet('{file_path}')
                WHERE {self.split_cond}
            """).df()
            if df.empty:
                continue

            inputs = torch.from_numpy(df[self.input_features].to_numpy(dtype="float32"))
            targets = torch.from_numpy(df[self.target_features].to_numpy(dtype="float32"))

            n_days = len(df) // self.bar_per_day
            inputs = inputs.reshape(n_days, self.bar_per_day, len(self.input_features))
            targets = targets.reshape(n_days,self.bar_per_day, len(self.target_features))

            x = inputs.unfold(dimension=1, size=self.seq_len, step=1)   #* [days, windows, features, seq_len]
            x = x.permute(0, 1, 3, 2)                                   #* [days, windows, seq_len, features]

            y = targets[:, self.step:]

            y = y.unfold(dimension=1,size=self.seq_len,step=1)
            y = y.permute(0, 1, 3, 2)

            n_windows = min(x.shape[1],y.shape[1])
            x = x[:, :n_windows]
            y = y[:, :n_windows]

            x = x.reshape(-1,self.seq_len,len(self.input_features))
            y = y.reshape(-1,self.seq_len,len(self.target_features))

            for i in range(len(x)):
                yield x[i], y[i]

            del df, inputs, targets, x, y

        con.close()