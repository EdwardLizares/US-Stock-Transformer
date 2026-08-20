import os
import torch
import pandas as pd

from pathlib import Path

from model_training import eval_loss

def process_losses(losses: list[dict], key = "MAE Loss"):
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
    header = f"{' '*20}"+"".join(f"{col_name:<{cs}}" for col_name in col_names)+"\n"
    rows = "".join(
        f"{row_name}: {parameters[i]}\n"
        f"    Training:       {tensor_to_string(losses[i*3], cs)}  >  {losses[i*3].mean():.4f}\n"
        f"    Validation:     {tensor_to_string(losses[i*3+1], cs)}  >  {losses[i*3+1].mean():.4f}\n"
        f"    Testing:        {tensor_to_string(losses[i*3+2], cs)}  >  {losses[i*3+2].mean():.4f}\n"
        f"    "
        f"\n"
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

def store_result(output_path, new_data):
    os.makedirs("results", exist_ok=True)
    df = pd.read_parquet(output_path) if Path(output_path).exists() else pd.DataFrame(
        columns = [
            "model",
            "bar_width",
            "train",
            "val",
            "test",
            "epoch",
            "file_limit"
        ]
    )
    confirm = ""
    while confirm not in ('y', 'n'):
        print(new_data)
        confirm = input("Enter this data? (y/n)")
        if confirm == 'y':
            confirm = input("Are you sure? (y/n)")
            if confirm == 'y':
                df.loc[len(df)] = new_data
            else:
                break
        else:
            break
    df.to_parquet(output_path)

def process_result(model, train_val, test_loss, epoch, file_limit):
    """
    Outputs a list of dictionaries
    """
    return {
        "model": model.cfg["name"],
        "bar_width": model.cfg["name"].split('-')[1][1:],
        "train": train_val[0],
        "val": train_val[1],
        "test": test_loss,
        "epoch": epoch,
        "file_limit": file_limit,
    }
