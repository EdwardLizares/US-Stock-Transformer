import os
import torch

from tqdm import tqdm
from setup import DGF

def get_dist(mean, std, dgf = DGF):
    if dgf is None:
        dist = torch.distributions.Normal(mean, std)
    else:
        dist = torch.distributions.StudentT(dgf, mean, std)
    return dist

def batch_loss(x, y, model, dgf = DGF):
    """
    Assume x and y are on the correct device already \n
    Returns NLL for back propagation
    """
    mean, std = model(x)
    dist = get_dist(mean, std, dgf)
    y_norm = (y - model.target_mean) / model.target_std

    return -dist.log_prob(y_norm).mean()

def eval_loss(data_loader, model, device, max_batches = float("inf"), pbar = None, desc="") -> dict: 
    """
    Returns a dict of MAE loss and Negative Log Loss
    """
    num_batches = min(len(data_loader), max_batches)
    avg_mae = 0
    avg_nll = 0
    avg_std = 0
    avg_z2 = 0

    for i, (p, t) in enumerate(data_loader):
        if i == num_batches:
            break

        p = p.to(device, non_blocking=True)
        t = t.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            mean, std = model(p)

            t_norm = (t - model.target_mean) / model.target_std

            #* MAE per target column
            mae = torch.abs(mean - t_norm).mean(dim=(0, 1))

            #* NLL per target column
            dist = get_dist(mean, std)
            nll = -dist.log_prob(t_norm).mean(dim=(0, 1))

            #* Predicted STD per target column
            col_std = std.mean(dim=(0, 1))

            #* Z statistics
            z = (t_norm - mean) / std.clamp_min(1e-6)
            z2 = z.pow(2).mean(dim=(0, 1))

        avg_mae += (mae - avg_mae) / (i+1)
        avg_nll += (nll - avg_nll) / (i+1)
        avg_std += (col_std - avg_std) / (i+1)
        avg_z2 += (z2 - avg_z2) / (i+1)

        if pbar is not None:
            pbar.update(1)
            if i % max(1,int(num_batches*0.001))==0:
                pbar.set_description(f"{desc} ({i}/{num_batches}) [{pbar.n}/{pbar.total}]")
    return {"NLL": avg_nll, "STD": avg_std, "MAE": avg_mae, "Z^2": avg_z2}

def load_model(path, model, device, optimizer=None, cuda_scaler=None, scheduler=None):
    checkpoint = torch.load(path, map_location=device)
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    print(missing, unexpected)
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if cuda_scaler is not None and "cuda_scaler" in checkpoint:
        cuda_scaler.load_state_dict(checkpoint["cuda_scaler"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint

def evaluate_model(train_dl, val_dl, model, device, eval_bs, pbar = None):
    """
    Returns a list of dictionaries, with each dictionary correspoding to a function in eval_fns
    """
    with torch.inference_mode():
        train_metrics = eval_loss(train_dl, model, device, eval_bs, pbar,
                                    desc="Evaluating model on training data...")
        val_metrics = eval_loss(val_dl, model, device, eval_bs, pbar,
                                  desc="Evaluating model on validation data...")    
    return train_metrics, val_metrics

def evaluate_best_model(model, device, optimizer, cuda_scaler, scheduler, train_dl, val_dl,
                        eval_bs, pbar = None, reevaluate = False):
    if os.path.exists(model.best_path):
        checkpoint = load_model(model.best_path, model, device)
        if reevaluate is False:
            return checkpoint["train_losses"][-1], checkpoint["val_losses"][-1]
        else:
            return evaluate_model(train_dl, val_dl, model, device, eval_bs, pbar)
    else:
        raise FileNotFoundError("Best parameters of the model could not be found")

def train_model_cuda(model, device, optimizer, cuda_scaler, scheduler, max_epochs,
                     train_dl, val_dl, eval_bs):
    #* LOADS MODEL
    if os.path.exists(model.checkpoint_path):
        print("Continuing from previous checkpoint...")
        checkpoint = load_model(model.checkpoint_path, model, device, optimizer, cuda_scaler, scheduler)
        bvm, epoch, train_losses, val_losses = (
            checkpoint["bvm"], checkpoint["epoch"]+1, checkpoint["train_losses"], checkpoint["val_losses"]
        )
    elif os.path.exists(model.best_path):
        print("Continuing from best parameter state...")
        checkpoint = load_model(model.best_path, model, device, optimizer, cuda_scaler, scheduler)
        bvm, epoch, train_losses, val_losses = (
            checkpoint["bvm"], checkpoint["epoch"]+1, checkpoint["train_losses"], checkpoint["val_losses"]
        ) #* Does at most an additional 3 checkpoints when checkpoint path file is deleted and best exists
    else:
        bvm, epoch, train_losses, val_losses = float("inf"), 0, [], []

    eval_steps = min(eval_bs, len(train_dl)) + min(eval_bs, len(val_dl))
    pbar = tqdm(total=(max_epochs-epoch)*(len(train_dl)+eval_steps), desc=f"Setting up...".ljust(80),
                bar_format="|{bar}| {percentage:3.1f}% ({elapsed}) {desc}", position=0, leave=False, delay=0.5)
    pbar.write((f"Epoch {epoch+1}:\n"))
    try:
        for epoch in range(epoch, max_epochs):
            pbar.write(f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}\n")
            #* TRAINS MODEL
            model.train()
            for x, y in train_dl:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(device_type="cuda",dtype=torch.float16):
                    loss = batch_loss(x, y, model)
                cuda_scaler.scale(loss).backward()
                cuda_scaler.step(optimizer)
                cuda_scaler.update()

                pbar.update(1)
                if (pbar.n % max(1,int(pbar.total*0.001))==0):
                    pbar.set_description(f"Training {model.cfg['name']}... [{pbar.n}/{pbar.total}]")

            #* EVALUATES MODEL
            model.eval()
            pbar.set_description(f"Evaluating Epoch {epoch}... [{pbar.n}/{pbar.total}]")
            train_metrics, val_metrics = evaluate_model(train_dl, val_dl, model, device, eval_bs, pbar)
            pbar.write((
                        f"Epoch {epoch+1}:\n"
                        f"Training Loss:\n"
                        f"   (MAE) {train_metrics['MAE'].mean()}\n"
                        f"   (NLL) {train_metrics['NLL'].mean()}\n"
                        f"Validation Loss:\n"
                        f"   (MAE) {val_metrics['MAE'].mean()}\n"
                        f"   (NLL) {val_metrics['NLL'].mean()}\n"))
            train_losses.append(train_metrics)
            val_losses.append(val_metrics)

            #* CHECKS SCORE
            cvm = val_metrics['NLL'].mean().item()
            scheduler.step(cvm)

            #* SAVES MODEL
            checkpoint = {
                "model": model.state_dict(),
                "cfg": model.cfg,
                "optimizer": optimizer.state_dict(),
                "cuda_scaler": cuda_scaler.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "bvm": bvm
            }
            if (cvm < bvm):
                bvm = cvm
                checkpoint["bvm"] = cvm
                torch.save(checkpoint, model.best_path)
            pbar.write((f"Best Validation: {bvm}\n{'-'*100}\n"))
            torch.save(checkpoint, model.checkpoint_path)
    finally:
            pbar.close()
    print("Finished")
    return train_losses, val_losses

def model_setup(model_cls, cfg, train_norms, device, optimizer_cls, lr, weight_decay, scaler_cls, scale_type):
    model = model_cls(cfg, train_norms)
    model.to(device)
    model_params = sum(p.numel() for p in model.parameters())
    print(model_params)
    optimizer = optimizer_cls(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = scaler_cls(scale_type)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( #! HARD CODED
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-6
    )
    return model, model_params, optimizer, scaler, scheduler
