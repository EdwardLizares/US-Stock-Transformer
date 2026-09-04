import torch
import torch.nn.functional as func

#from setup import StockGPT_cfg as cfg

class MultiheadAttention(torch.nn.Module):
    """
    Creates a wide casual attention matrix and splits it
    """
    def __init__(self, in_dim, out_dim, num_heads, qkv_bias, mx_sql):
        super().__init__()
        assert (out_dim % num_heads == 0)
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.W_q = torch.nn.Linear(in_dim, out_dim, qkv_bias)
        self.W_k = torch.nn.Linear(in_dim, out_dim, qkv_bias)
        self.W_v = torch.nn.Linear(in_dim, out_dim, qkv_bias)
        self.out_proj = torch.nn.Linear(out_dim, out_dim)

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
        self.ln1 = LayerNorm(cfg["output_dim"])
        self.mha = MultiheadAttention(cfg["output_dim"], cfg["output_dim"], cfg["n_heads"],
                                      cfg["qkv_bias"], cfg["seq_len"])
        self.ln2 = LayerNorm(cfg["output_dim"])
        self.ff = FeedForward(cfg["output_dim"])

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class StockBPT(torch.nn.Module):
    def __init__(self, cfg, train_norms = None, print_norms = True):
        super().__init__()
        self.cfg = cfg
        self.checkpoint_path = cfg["checkpoint_path"]
        self.best_path = cfg["best_path"]
        self.input_proj = torch.nn.Linear(len(cfg["input_features"]), cfg["output_dim"])
        self.pos_emb = torch.nn.Embedding(cfg["seq_len"], cfg["output_dim"])
        self.transformer_blocks = torch.nn.Sequential(
            *[StockTransformer(cfg) for _ in range(cfg["n_transformers"])]
        )
        self.final_norm = LayerNorm(cfg["output_dim"])
        #self.std_head = torch.nn.Sequential(
        #    torch.nn.Linear(cfg["output_dim"], cfg["output_dim"]//4, False),
        #    torch.nn.GELU(),
        #    torch.nn.Linear(cfg["output_dim"]//4, len(cfg["target_features"]), False),
        #)
        #self.mean_head = torch.nn.Linear(cfg["output_dim"], len(cfg["target_features"]), False)
        self.out_head = torch.nn.Linear(cfg["output_dim"], 2*len(cfg["target_features"]), False)

        if train_norms is None:
            train_norms = (
                torch.zeros(len(cfg["input_features"])),
                torch.ones(len(cfg["input_features"])),
                torch.zeros(len(cfg["target_features"])),
                torch.ones(len(cfg["target_features"])),
            )

        self.register_buffer("input_mean", train_norms[0])
        self.register_buffer("input_std", train_norms[1])
        self.register_buffer("target_mean", train_norms[2])
        self.register_buffer("target_std", train_norms[3])

        if print_norms:
            print((f"Input Norm: {self.input_mean.size()}|{self.input_std.size()}\n"
                   f"Target Norm: {self.target_mean.size()}|{self.target_std.size()}"))

    def forward(self, x):
        _, sql, _ = x.shape            #! This is for later making predictions off bs=1, sql<25
        x = ( x - self.input_mean ) / self.input_std
        proj = self.input_proj(x)
        pos_emb = self.pos_emb(torch.arange(sql, device=x.device))
        x = proj + pos_emb
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        #mean = self.mean_head(x)
        #raw_std = self.std_head(x)
        x = self.out_head(x)
        mean, raw_std = x.chunk(2, dim=-1)
        std = torch.nn.functional.softplus(raw_std) + 1e-6
        return mean, std

class LinearModel(torch.nn.Module):
    """
    Single Linear Layer
    """
    def __init__(self, cfg, train_norms):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("input_mean", train_norms[0])
        self.register_buffer("input_std", train_norms[1])
        self.register_buffer("target_mean", train_norms[2])
        self.register_buffer("target_std", train_norms[3])
        self.checkpoint_path = cfg["checkpoint_path"]
        self.best_path = cfg["best_path"]        
        self.linear_layer = torch.nn.Linear(len(cfg["input_features"]), cfg["output_dim"])
        self.out_head = torch.nn.Linear(cfg["output_dim"], 2*len(cfg["target_features"]), False)

    def forward(self, x):
        x = ( x - self.input_mean ) / self.input_std
        x = self.linear_layer(x)
        x = self.out_head(x)
        mean, raw_std = x.chunk(2, dim=-1)
        std = torch.nn.functional.softplus(raw_std) + 1e-6
        return mean, std

class NaiveModel(torch.nn.Module):
    def __init__(self, cfg, train_norms):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("input_mean", train_norms[0])
        self.register_buffer("input_std", train_norms[1])
        self.register_buffer("target_mean", train_norms[2])
        self.register_buffer("target_std", train_norms[3])
        self.target_indices = [
            cfg["input_features"].index(feature)
            for feature in cfg["target_features"]
        ]

    def forward(self, x):
        x = (x - self.input_mean) / self.input_std
        mean = x[:, :, self.target_indices]
        std = torch.ones_like(mean)
        return mean, std  #* Drops columns from input features I don't need to predict

if __name__ == "__main__":
    naive = LinearModel(cfg, [torch.ones(36), torch.zeros(36), torch.ones(36), torch.zeros(36)])
    test_data = torch.rand(1, 36, 36)
    print(test_data)
    print(test_data.shape)
    test_predict = naive(test_data)
    print(test_predict.shape)