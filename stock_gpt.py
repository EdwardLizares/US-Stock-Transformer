import pandas as pd
import torch
import torch.nn.functional as func

from torch.utils.data import Dataset

from setup import Stock_GPT_cfg as C

class StockDataset(Dataset):
    def __init__(self, df: pd.DataFrame, seq_len =  C["SEQ_LEN"], bar_count = C["BAR_PER_DAY"],
                 input_features = C["INPUT_FEATURES"], target_features = C["TARGET_FEATURES"]):
        """
        Assumes entire provided df DataFrame is the dataset and is normalized
        """
        self.input = df[input_features].to_numpy(dtype="float32")
        self.target = df[target_features].to_numpy(dtype="float32")
        self.seq_len = seq_len
        self.bar_count = bar_count  # bars in a day (15min --> 26)

    def __len__(self):
        return len(self.input)//self.bar_count

    def __getitem__(self, idx: int):
        """
        Returns both the input and target OLHC sequence
        """
        idx*=self.bar_count
        return (torch.from_numpy(self.input[idx:idx+self.seq_len]),
                torch.from_numpy(self.target[idx+1:idx+self.seq_len+1]))

class MultiheadAttention(torch.nn.Module):
    """
    Creates a wide casual attention matrix and splits it
    """
    def __init__(self, in_dim, out_dim, num_heads = C["N_HEADS"],
                 qkv_bias = C["QKV_BIAS"], mx_sql = C["SEQ_LEN"]):
        super().__init__()
        assert (out_dim % num_heads == 0)
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.W_q = torch.nn.Linear(in_dim, out_dim, qkv_bias)
        self.W_k = torch.nn.Linear(in_dim, out_dim, qkv_bias)
        self.W_v = torch.nn.Linear(in_dim, out_dim, qkv_bias)
        self.out_proj = torch.nn.Linear(out_dim, out_dim)
        self.register_buffer("c_mask", torch.triu(torch.ones(mx_sql, mx_sql), diagonal=1))

    def forward(self, x):
        bs, sql, _ = x.shape            #! This is for later making predictions off bs=1, sql<25

        qs = (self.W_q(x)).view(
            bs, sql, self.num_heads, self.head_dim
            ).transpose(1, 2)
        ks = (self.W_k(x)).view(
            bs, sql, self.num_heads, self.head_dim
            ).transpose(1, 2)
        vs = (self.W_v(x)).view(
            bs, sql, self.num_heads, self.head_dim
            ).transpose(1, 2)

        context = func.scaled_dot_product_attention(qs, ks, vs, is_causal=True)
        return self.out_proj(context.transpose(1, 2).contiguous().view(bs, sql, self.out_dim))

class LayerNorm(torch.nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(out_dim))
        self.shift = torch.nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x-mean)/torch.sqrt(var+1e-5)
        return self.scale * x_norm + self.shift

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
    def __init__(self, cfg, train_norms, print_norms = False):
        super().__init__()
        self.cfg = cfg
        self.save_path = cfg["SAVE_PATH"]
        self.input_proj = torch.nn.Linear(len(cfg["INPUT_FEATURES"]), cfg["OUTPUT_DIM"])
        self.pos_emb = torch.nn.Embedding(cfg["SEQ_LEN"], cfg["OUTPUT_DIM"])
        self.transformer_blocks = torch.nn.Sequential(
            *[StockTransformer(cfg) for _ in range(cfg["N_TRANSFORMERS"])]
        )
        self.final_norm = LayerNorm(cfg["OUTPUT_DIM"])
        self.out_head = torch.nn.Linear(cfg["OUTPUT_DIM"], len(cfg["TARGET_FEATURES"]), False)
        self.register_buffer("input_mean", train_norms[0])
        self.register_buffer("input_std", train_norms[1])
        self.register_buffer("target_mean", train_norms[2])
        self.register_buffer("target_std", train_norms[3])
        if print_norms:
            print((f"Input Norm: {self.input_mean}|{self.input_std}\n"
                   f"Target Norm: {self.target_mean}|{self.target_std}"))

    def forward(self, x):
        bs, sql, _ = x.shape            #! This is for later making predictions off bs=1, sql<25
        x = ( x - self.input_mean ) / self.input_std
        proj = self.input_proj(x)
        pos_emb = self.pos_emb(torch.arange(sql, device=x.device))
        x = proj + pos_emb
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        return self.out_head(x)

class NaiveGPT(torch.nn.Module):
    """
    Single Linear Layer
    """
    def __init__(self, cfg, train_norms):
        super().__init__()
        self.register_buffer("input_mean", train_norms[0])
        self.register_buffer("input_std", train_norms[1])
        self.register_buffer("target_mean", train_norms[2])
        self.register_buffer("target_std", train_norms[3])    
        self.linear_layer = torch.nn.Linear(len(cfg["INPUT_FEATURES"]), cfg["OUTPUT_DIM"])
        self.out_head = torch.nn.Linear(cfg["OUTPUT_DIM"], len(cfg["TARGET_FEATURES"]), False)

    def forward(self, x):
        bs, sql, _ = x.shape
        x = self.linear_layer(x)
        x = self.out_head(x)
        return x

if __name__ == "__main__":
    naive = NaiveGPT(C, [torch.ones(25), torch.zeros(25), torch.ones(25), torch.zeros(25)])
    test_data = torch.rand(1, 25, 13)
    print(test_data)
    print(test_data.shape)
    test_predict = naive(test_data)
    print(test_predict.shape)
