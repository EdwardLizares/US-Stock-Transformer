import torch
import duckdb

from torch.utils.data import IterableDataset

from setup import StockGPT_cfg as cfg

class StockDataset(IterableDataset):
    def __init__(self, con, view_name: str, dq: duckdb.DuckDBPyRelation, seq_len: int =  cfg["seq_len"],
                 input_features: str = cfg["input_features"], target_features: str = cfg["target_features"],
                 step: int = cfg["step"]):
        """
        Converts a duckdb relation object into a dataset
        """
        self.con = con
        self.view_name = view_name
        dq.create_view(view_name)

        self.input_features = input_features
        self.target_features = target_features

        self.seq_len = seq_len
        self.step = step

    def __iter__(self):
        data = self.con.execute(f"""
            SELECT *
            FROM {self.view_name}
            ORDER BY T_1, date, bar
        """)

        while True:
            chunk = data.fetch_df_chunk(100)
            if chunk.empty:
                break

            for (_, _), day in chunk.groupby(["T_1", "date"],sort=False):
                inputs = day[self.input_features].to_numpy(dtype="float32")
                targets = day[self.target_features].to_numpy(dtype="float32")

                max_start = len(day) - self.seq_len - self.step + 1
                for start in range(max_start):
                    input_start = start
                    input_end = start + self.seq_len
                    target_start = start + self.step
                    target_end = target_start + self.seq_len

                    x = inputs[input_start:input_end]
                    y = targets[target_start:target_end]

                    yield (torch.from_numpy(x),torch.from_numpy(y))
