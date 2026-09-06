import os
import torch

from tqdm import tqdm

def batch_loss(model, x, y):
    logits = model(x)
    weights = torch.tensor(model.cfg["target_weights"], device=logits.device)
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, len(model.cfg["target_features"])),
        y.reshape(-1),
        weight = weights
    )

"""
def batch_loss(model, x, y, alpha=1.0):
    logits = model(x)
    logits_flat = logits.reshape(-1, len(model.cfg["target_features"]))
    y_flat = y.reshape(-1)

    ce = torch.nn.functional.cross_entropy(logits_flat, y_flat)

    probs = torch.softmax(logits_flat, dim=-1)

    p_down = probs[:, 0]
    p_up = probs[:, 2]

    y_down = (y_flat == 0).float()
    y_up = (y_flat == 2).float()

    down_tp = (p_down * y_down).sum()
    down_fp = (p_down * (1 - y_down)).sum()
    up_tp = (p_up * y_up).sum()
    up_fp = (p_up * (1 - y_up)).sum()

    down_precision = down_tp / (down_tp + down_fp + 1e-8)
    up_precision = up_tp / (up_tp + up_fp + 1e-8)

    precision_loss = 1 - (down_precision + up_precision) / 2

    return ce + alpha * precision_loss
"""

def eval_loss(data_loader, model, device, max_batches=float("inf"), pbar=None, desc="", alpha=1.0):    
    """
    Returns total loss, CE, precision loss, accuracy, precision, recall, and F1
    """
    num_batches = min(len(data_loader), max_batches)
    avg_loss = 0
    avg_ce = 0
    avg_precision_loss = 0
    correct = 0
    total = 0
    n_classes = len(model.cfg["target_features"])
    tp = torch.zeros(n_classes, device=device)
    fp = torch.zeros(n_classes, device=device)
    fn = torch.zeros(n_classes, device=device)

    for i, (p, t) in enumerate(data_loader):
        if i == num_batches:
            break
        p = p.to(device, non_blocking=True)
        t = t.to(device, non_blocking=True).long()

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(p)

        logits = logits.float()
        logits_flat = logits.reshape(-1, n_classes)
        t_flat = t.reshape(-1)

        ce = torch.nn.functional.cross_entropy(logits_flat, t_flat)

        probs = torch.softmax(logits_flat, dim=-1)
        p_down = probs[:, 0]
        p_up = probs[:, 2]
        y_down = (t_flat == 0).float()
        y_up = (t_flat == 2).float()

        down_tp_soft = (p_down * y_down).sum()
        down_fp_soft = (p_down * (1 - y_down)).sum()
        up_tp_soft = (p_up * y_up).sum()
        up_fp_soft = (p_up * (1 - y_up)).sum()

        down_precision_soft = down_tp_soft / (down_tp_soft + down_fp_soft + 1e-8)
        up_precision_soft = up_tp_soft / (up_tp_soft + up_fp_soft + 1e-8)

        precision_loss = 1 - (down_precision_soft + up_precision_soft) / 2
        loss = ce + alpha * precision_loss

        preds = logits.argmax(dim=-1)
        correct += (preds == t).sum()
        total += t.numel()

        for c in range(n_classes):
            tp[c] += ((preds == c) & (t == c)).sum()
            fp[c] += ((preds == c) & (t != c)).sum()
            fn[c] += ((preds != c) & (t == c)).sum()

        avg_loss += (loss - avg_loss) / (i+1)
        avg_ce += (ce - avg_ce) / (i+1)
        avg_precision_loss += (precision_loss - avg_precision_loss) / (i+1)

        if pbar is not None:
            pbar.update(1)
            if i % max(1, int(num_batches*0.001)) == 0:
                pbar.set_description(f"{desc} ({i}/{num_batches}) [{pbar.n}/{pbar.total}]")

    accuracy = correct / total
    precision = tp / (tp + fp).clamp_min(1)
    recall = tp / (tp + fn).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-8)

    return {
        "LOSS": avg_loss,
        "CE": avg_ce,
        "PREC_LOSS": avg_precision_loss,
        "ACC": accuracy,
        "PREC": precision,
        "REC": recall,
        "F1": f1
    }

def precision_recall_curve(data_loader, model, device, cls=2):
    probs_all, targets_all = [], []

    model.eval()
    with torch.inference_mode():
        for x, y in data_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).long()

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x)

            probs = torch.softmax(logits.float(), dim=-1)[..., cls]
            probs_all.append(probs.flatten())
            targets_all.append((y == cls).flatten())

    probs = torch.cat(probs_all)
    targets = torch.cat(targets_all)

    for threshold in torch.arange(0.1, 1.0, 0.05, device=device):
        pred = probs >= threshold
        tp = (pred & targets).sum()
        fp = (pred & ~targets).sum()
        fn = (~pred & targets).sum()

        precision = tp / (tp + fp).clamp_min(1)
        recall = tp / (tp + fn).clamp_min(1)

        print(
            f"{threshold.item():.2f} | "
            f"PREC {precision.item():.4f} | "
            f"REC {recall.item():.4f} | "
            f"N {pred.sum().item()}"
        )

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
                y = y.to(device, non_blocking=True).long()
                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(device_type="cuda",dtype=torch.float16):
                    loss = batch_loss(model, x, y)

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
                        f"   (CE)   {train_metrics['CE']}\n"
                        f"   (ACC)  {train_metrics['ACC']}\n"
                        f"   (PREC) {train_metrics['PREC']}\n"
                        f"   (REC)  {train_metrics['REC']}\n"
                        f"   (F1)   {train_metrics['F1']}\n"
                        f"Validation Loss:\n"
                        f"   (CE)   {val_metrics['CE']}\n"
                        f"   (ACC)  {val_metrics['ACC']}\n"
                        f"   (PREC) {val_metrics['PREC']}\n"
                        f"   (REC)  {val_metrics['REC']}\n"
                        f"   (F1)   {val_metrics['F1']}\n"))

            train_losses.append(train_metrics)
            val_losses.append(val_metrics)

            #* CHECKS SCORE
            cvm = val_metrics["CE"].item()
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
