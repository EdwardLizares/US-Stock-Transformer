import os
import torch
import pandas as pd

from pathlib import Path

from model_training_mpt import eval_loss

def process_losses(losses: list[dict], key = "CE"):
    return [loss_dict[key] for loss_dict in losses]

def tensor_to_string(t, cs):
    return "".join(f"{v.item():<{cs}.4f}" for v in t)

def format_num(n):
    if n >= 1e9:
        return f"{n / 1e9:.1f}B"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)

def print_loss_analysis(losses, model_names, parameters, col_names, key, cs = 9):
    title = f"{key}\n"
    bound = f"\n{'-'*110}\n\n"

    if key in ["CE", "ACC"]:
        header = ""
        rows = "".join(
            f"{row_name}: {parameters[i]}\n"
            f"    Training:       {losses[i*3].item():.4f}\n"
            f"    Validation:     {losses[i*3+1].item():.4f}\n"
            f"    Testing:        {losses[i*3+2].item():.4f}\n"
            f"    \n"
        for i, row_name in enumerate(model_names))
    else:
        header = f"{' '*20}"+"".join(f"{col_name:<{cs}}" for col_name in col_names)+"\n"
        rows = "".join(
            f"{row_name}: {parameters[i]}\n"
            f"    Training:       {tensor_to_string(losses[i*3], cs)}  >  {losses[i*3].mean():.4f}\n"
            f"    Validation:     {tensor_to_string(losses[i*3+1], cs)}  >  {losses[i*3+1].mean():.4f}\n"
            f"    Testing:        {tensor_to_string(losses[i*3+2], cs)}  >  {losses[i*3+2].mean():.4f}\n"
            f"    \n"
        for i, row_name in enumerate(model_names))

    output = [
        bound,
        title,
        bound,
        header,
        rows,
        bound
    ]
    print("".join(output))

def test_model(test_dl, model, device, eval_bs, pbar=None):
    """
    Evaluates model on testing data only.
    """
    model.eval()
    with torch.no_grad():
        test_metrics = eval_loss(test_dl, model, device, eval_bs, pbar, desc="Evaluating model on testing data...")
    return (test_metrics,)

def process_metrics(metrics):
    return {
        key: value.detach().cpu().tolist()
        for key, value in metrics.items()
    }