import torch

from pathlib import Path

from setup import StockGPT_cfg
from stock_gpt import StockGPT
from dataloader_builder import calculate_training_norms

def setup_model(source_model: Path):
    assert Path(source_model).exists(), "Must be a Path to an existing model"
    cuda = True if torch.cuda.is_available() else False
    print("CUDA available:", cuda)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    device = torch.device("cuda" if cuda else "cpu")

    model = StockGPT(StockGPT_cfg)
    best = torch.load(source_model, map_location=device)
    model.load_state_dict(best["model"])
    model.to(device)
    model.eval()

    return model, device

def query_model(model, x):
    device = next(model.parameters()).device
    x = x.to(device)

    with torch.inference_mode():
        mean, std = model(x)
        mean = mean * model.target_std + model.target_mean
        std = std * model.target_std
    return mean, std

if __name__ == "__main__":
    model, device = setup_model("model_parameters/checkpoint_stock_gpt_5min")
    mean, std = query_model(model, torch.randn(1, 77, len(StockGPT_cfg["input_features"])))
    print(mean, std)