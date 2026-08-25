import torch

from pathlib import Path

from setup import StockGPT_cfg
from stock_gpt import StockGPT

def setup_model(source_model: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best = torch.load(source_model, map_location=device)
    cfg = best["cfg"]
    model = StockGPT(cfg, None, False)
    model.load_state_dict(best["model"])
    model.to(device)
    model.eval()

    return model, device

def query_model(model, x):
    """
    Expects (bs, sql, 13)
    Returns all predicted next column per ticker in batch
    """
    device = next(model.parameters()).device
    x = x.to(device)

    with torch.inference_mode():
        mean, std = model(x)
        mean = mean * model.target_std + model.target_mean
        std = std * model.target_std

    seq_len = x.shape[1]

    return x, mean, std

def print_prediction(prev, res, std, input_features, target_features):

    if prev is None or res is None or std is None:
        print("No prediction available.")
        return
    
    prev = prev.detach().cpu().squeeze()
    res = res.detach().cpu().squeeze()
    std = std.detach().cpu().squeeze()

    target_indices = [input_features.index(feature) for feature in target_features]

    prev = prev[target_indices]
    print(f"{'Feature':<10} {'Previous':>10} {'Residual':>10} {'Std':>10}")
    print("-" * 44)
    for feature, p, m, s in zip(target_features, prev, res, std):
        print(
            f"{feature:<10}"
            f"{p.item():>10.4f}"
            f"{m.item():>10.4f}"
            f"{s.item():>10.4f}"
        )
if __name__ == "__main__":
    model, device = setup_model("model_parameters/checkpoint_stock_gpt_1min_residuals")
    prev, mean, std = query_model(model, torch.randn(2, 389, len(StockGPT_cfg["input_features"])))
    print(mean, std)
    print(mean.shape)