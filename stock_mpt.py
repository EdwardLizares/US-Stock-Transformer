import torch
import torch.nn.functional as func

#from setup import StockGPT_cfg as cfg

class RMSNorm(torch.nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        return self.weight * x * torch.rsqrt(rms + self.eps)

class SwiGLU(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        hidden = int((8*dim/3 + 255)//256*256)
        self.w1 = torch.nn.Linear(dim, hidden, bias=False)
        self.w2 = torch.nn.Linear(dim, hidden, bias=False)
        self.w3 = torch.nn.Linear(hidden, dim, bias=False)
    def forward(self, x):
        return self.w3(func.silu(self.w1(x)) * self.w2(x))

def rotate_half(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)

class RoPE(torch.nn.Module):
    def __init__(self, head_dim, max_seq_len=390, base=10000):
        super().__init__()
        assert head_dim % 2 == 0
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_seq_len).float()
        freqs = torch.outer(positions, inv_freq)
        emb = torch.repeat_interleave(freqs, 2, dim=-1)
        self.register_buffer("cos", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin", emb.sin()[None, None, :, :], persistent=False)
    def forward(self, q, k):
        sql = q.shape[-2]
        cos = self.cos[:, :, :sql].to(dtype=q.dtype)
        sin = self.sin[:, :, :sql].to(dtype=q.dtype)
        q = q*cos + rotate_half(q)*sin
        k = k*cos + rotate_half(k)*sin
        return q, k

class MultiheadAttention(torch.nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, qkv_bias, seq_len):
        super().__init__()
        assert out_dim % num_heads == 0
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        assert self.head_dim % 2 == 0
        self.W_q = torch.nn.Linear(in_dim, out_dim, qkv_bias)
        self.W_k = torch.nn.Linear(in_dim, out_dim, qkv_bias)
        self.W_v = torch.nn.Linear(in_dim, out_dim, qkv_bias)
        self.out_proj = torch.nn.Linear(out_dim, out_dim)
        self.rope = RoPE(self.head_dim, seq_len)
    def forward(self, x):
        bs, sql, _ = x.shape
        qs = self.W_q(x).view(bs, sql, self.num_heads, self.head_dim).transpose(1, 2)
        ks = self.W_k(x).view(bs, sql, self.num_heads, self.head_dim).transpose(1, 2)
        vs = self.W_v(x).view(bs, sql, self.num_heads, self.head_dim).transpose(1, 2)
        qs, ks = self.rope(qs, ks)
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
        self.norm1 = RMSNorm(cfg["output_dim"])
        self.mha = MultiheadAttention(cfg["output_dim"], cfg["output_dim"],
                                      cfg["n_heads"], cfg["qkv_bias"], cfg["seq_len"])
        self.norm2 = RMSNorm(cfg["output_dim"])
        self.ff = SwiGLU(cfg["output_dim"])
    def forward(self, x):
        x = x + self.mha(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

class StockMPT(torch.nn.Module):
    def __init__(self, cfg, train_norms = None):
        super().__init__()
        self.cfg = cfg
        self.pm_bars = cfg["pm_bars"]
        self.checkpoint_path = cfg["checkpoint_path"]
        self.best_path = cfg["best_path"]
        self.input_proj = torch.nn.Linear(len(cfg["input_features"]), cfg["output_dim"])
        self.transformer_blocks = torch.nn.Sequential(
            *[StockTransformer(cfg) for _ in range(cfg["n_transformers"])]
        )
        self.final_norm = RMSNorm(cfg["output_dim"])
        self.out_head = torch.nn.Linear(cfg["output_dim"], len(cfg["target_features"]), False)

        if train_norms is None:
            train_norms = (
                torch.zeros(len(cfg["input_features"])),
                torch.ones(len(cfg["input_features"])),
            )

        self.register_buffer("input_mean", train_norms[0])
        self.register_buffer("input_std", train_norms[1])

    def forward(self, x):
        _, sql, _ = x.shape            #! This is for later making predictions off bs=1, sql<25
        x = ( x - self.input_mean ) / self.input_std
        x = self.input_proj(x)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        x = x[:, self.pm_bars:, :]
        return self.out_head(x)

class LinearModel(torch.nn.Module):
    """
    Single Linear Layer
    """
    def __init__(self, cfg, train_norms):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("input_mean", train_norms[0])
        self.register_buffer("input_std", train_norms[1])
        self.checkpoint_path = cfg["checkpoint_path"]
        self.best_path = cfg["best_path"]        
        self.linear_layer = torch.nn.Linear(len(cfg["input_features"]), cfg["output_dim"])
        self.out_head = torch.nn.Linear(cfg["output_dim"], len(cfg["target_features"]), False)

    def forward(self, x):
        x = (x - self.input_mean) / self.input_std
        x = self.linear_layer(x)
        x = x[:, self.cfg["pm_bars"]:, :]
        return self.out_head(x)

class NaiveModel(torch.nn.Module):
    def __init__(self, cfg, train_norms = None):
        super().__init__()
        self.cfg = cfg
        self.close_idx = cfg["input_features"].index("c")

    def forward(self, x):
        closes = x[:, :, self.close_idx]
        change = closes[:, 1:] - closes[:, :-1]

        logits = torch.zeros(x.size(0), x.size(1), 3, device=x.device)
        logits[:, 1:, 0][change <= -0.02] = 1
        logits[:, 1:, 1][change.abs() < 0.02] = 1
        logits[:, 1:, 2][change >= 0.02] = 1
        return logits[:, self.cfg["pm_bars"]:, :]