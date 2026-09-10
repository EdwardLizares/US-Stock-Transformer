import torch

class StockMPTE(torch.nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = torch.nn.ModuleList(models)

    def forward(self, x, return_uncertainty=False):
        probs = torch.stack([
            torch.softmax(model(x), dim=-1)
            for model in self.models
        ])

        mean_probs = probs.mean(dim=0)

        if return_uncertainty:
            std_probs = probs.std(dim=0)
            return mean_probs, std_probs

        return torch.log(mean_probs.clamp_min(1e-8))

@torch.no_grad()
def ensemble_uncertainty_analysis(dl, model, device, cls=2, prob_threshold=0.50):
    all_probs = []
    all_stds = []
    all_y = []

    model.eval()

    for x, y in dl:
        x = x.to(device)
        y = y.to(device)

        mean_probs, std_probs = model(x, return_uncertainty=True)

        all_probs.append(mean_probs[..., cls].flatten().cpu())
        all_stds.append(std_probs[..., cls].flatten().cpu())
        all_y.append(y.flatten().cpu())

    probs = torch.cat(all_probs)
    stds = torch.cat(all_stds)
    y = torch.cat(all_y)

    signal = probs >= prob_threshold
    probs = probs[signal]
    stds = stds[signal]
    correct = (y[signal] == cls)

    print(f"Signals: {len(probs)}")
    print(f"Precision: {correct.float().mean():.4f}")

    for threshold in [0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15]:
        keep = stds <= threshold
        n = keep.sum().item()

        if n:
            precision = correct[keep].float().mean().item()
            print(f"STD <= {threshold:.3f} | PREC {precision:.4f} | N {n}")