import pandas as pd
import torch

from torch.utils.data import Dataset

from setup import Stock_GPT_cfg as C

class StockDataset(Dataset):
    def __init__(self, df: pd.DataFrame, seq_len =  C["SEQ_LEN"],
                 bar_count = C["BAR_PER_DAY"]):
        """
        Assumes entire provided df DataFrame is the dataset and is normalized
        """
        self.data = df.to_numpy(dtype="float32")
        self.seq_len = seq_len
        self.bar_count = bar_count  # bars in a day (15min --> 26)

    def __len__(self):
        return len(self.data)//self.bar_count

    def __getitem__(self, idx: int):
        """
        Returns both the input and target OLHC sequence
        """
        idx*=self.bar_count
        return (torch.tensor(self.data[idx:idx+self.seq_len+1]),
                torch.tensor(self.data[idx+1:idx+self.seq_len+1]))

class MultiheadAttention(torch.nn.Module):
    """
    Creates a wide casual attention matrix and splits it
    """
    def __init__(self, in_dim, out_dim, num_heads = C["N_HEADS"],
                 qkv_bias = C["QKV_BIAS"], sql = C["SEQ_LEN"], bs = C["BATCH_SIZE"]):
        super().__init__()
        assert (out_dim % num_heads == 0)
        self.seq_len = sql
        self.batch_size = bs
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.W_q = torch.nn.Linear(in_dim, out_dim, qkv_bias)
        self.W_k = torch.nn.Linear(in_dim, out_dim, qkv_bias)
        self.W_v = torch.nn.Linear(in_dim, out_dim, qkv_bias)
        self.out_proj = torch.nn.Linear(out_dim, out_dim)
        self.register_buffer("c_mask", torch.triu(torch.ones(self.seq_len, self.seq_len), diagonal=1))

    def forward(self, x):
        qs = (self.W_q(x)).view(
            self.batch_size, self.seq_len, self.num_heads, self.head_dim
            ).transpose(1, 2)
        ks = (self.W_k(x)).view(
            self.batch_size, self.seq_len, self.num_heads, self.head_dim
            ).transpose(1, 2)
        vs = (self.W_v(x)).view(
            self.batch_size, self.seq_len, self.num_heads, self.head_dim
            ).transpose(1, 2)

        att_scores = qs @ ks.transpose(2, 3)
        att_scores.masked_fill_(self.c_mask.bool()[:self.seq_len, :self.seq_len], -torch.inf)
        att_weights = torch.softmax(att_scores / self.head_dim**0.5, dim=-1)

        return self.out_proj(
            (att_weights @ vs).transpose(1, 2).contiguous().view(self.batch_size, self.seq_len, self.out_dim)
        )

class LayerNorm(torch.nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(out_dim))
        self.shift = torch.nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x-mean)/torch.sqrt(var+1e-5)
        return self.scale * x_norm - self.shift

class FeedForward(torch.nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(out_dim, 4*out_dim),
            torch.nn.GELU(),
            torch.nn.Linear(4*out_dim, out_dim)
        )

    def forward(self, x):
        return self.layers(x)

class StockTransformer(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = LayerNorm(cfg["OUTPUT_DIM"])
        self.mha = MultiheadAttention(cfg["OUTPUT_DIM"], cfg["OUTPUT_DIM"], cfg["N_HEADS"])
        self.ln2 = LayerNorm(cfg["OUTPUT_DIM"])
        self.ff = FeedForward(cfg["OUTPUT_DIM"])

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class StockGPT(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.input_proj = torch.nn.Linear(cfg["HYPERPARAMETERS"], cfg["OUTPUT_DIM"])
        self.pos_emb = torch.nn.Embedding(cfg["SEQ_LEN"], cfg["OUTPUT_DIM"])
        self.transformer_blocks = torch.nn.Sequential(
            *[StockTransformer(cfg) for _ in range(cfg["N_TRANSFORMERS"])]
        )
        self.final_norm = LayerNorm(cfg["OUTPUT_DIM"])
        self.out_head = torch.nn.Linear(cfg["OUTPUT_DIM"], cfg["HYPERPARAMETERS"], False)

    def forward(self, batch):
        proj = self.input_proj(batch)
        pos_emb = self.pos_emb(torch.arange(self.cfg["SEQ_LEN"], device=batch.device))
        x = proj + pos_emb
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        return self.out_head(x)
