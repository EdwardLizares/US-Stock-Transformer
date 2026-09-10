import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path

CLASS_NAMES = ["DOWN", "FLAT", "UP"]

def get_day_table(dataset, idx):
    file_idx = np.searchsorted(dataset.offsets, idx, side="right") - 1
    local_idx = idx - dataset.offsets[file_idx]
    day_idx = local_idx // dataset.samples_per_day
    row_start = day_idx * dataset.total_bars

    dataset._load_file(file_idx)

    return dataset.cached_table.slice(
        row_start,
        dataset.total_bars
    ).to_pandas()

def safe_mean(series):
    x = pd.to_numeric(series, errors="coerce")
    return x.mean()

def safe_max(series):
    x = pd.to_numeric(series, errors="coerce")
    return x.max()

def describe_day(day, pm_bars):
    day = day.reset_index(drop=True)
    pm = day.iloc[:pm_bars]
    rth = day.iloc[pm_bars:]

    c = day["c"]
    h = day["h"]
    l = day["l"]
    v = day["v"]

    row = {
        "Tk": day["Tk"].iloc[0] if "Tk" in day.columns else None,
        "date": day["date"].iloc[0] if "date" in day.columns else None,

        "PRICE_OPEN": c.iloc[0],
        "PRICE_CLOSE": c.iloc[-1],
        "PRICE_MEAN": c.mean(),
        "PRICE_MIN": l.min(),
        "PRICE_MAX": h.max(),

        "RANGE_PCT": (
            (h.max() - l.min()) / l.min()
            if l.min() != 0 else np.nan
        ),

        "TREND_PCT": (
            c.iloc[-1] / c.iloc[0] - 1
            if c.iloc[0] != 0 else np.nan
        ),

        "TOTAL_VOLUME": v.sum(),
        "MEAN_VOLUME": v.mean(),
        "MAX_VOLUME": v.max(),
        "ZERO_VOLUME_FRAC": (v == 0).mean(),
    }

    if len(pm):
        row["PM_VOLUME"] = pm["v"].sum()

        row["PM_RANGE_PCT"] = (
            (pm["h"].max() - pm["l"].min()) / pm["l"].min()
            if pm["l"].min() != 0 else np.nan
        )

        row["PM_TREND_PCT"] = (
            pm["c"].iloc[-1] / pm["c"].iloc[0] - 1
            if pm["c"].iloc[0] != 0 else np.nan
        )

    if len(rth):
        row["RTH_VOLUME"] = rth["v"].sum()

        row["RTH_RANGE_PCT"] = (
            (rth["h"].max() - rth["l"].min()) / rth["l"].min()
            if rth["l"].min() != 0 else np.nan
        )

        row["RTH_TREND_PCT"] = (
            rth["c"].iloc[-1] / rth["c"].iloc[0] - 1
            if rth["c"].iloc[0] != 0 else np.nan
        )

    if "rv" in day.columns:
        row["MEAN_RV"] = safe_mean(day["rv"])
        row["MAX_RV"] = safe_max(day["rv"])

    if "ibkr_rv" in day.columns:
        row["MEAN_IBKR_RV"] = safe_mean(day["ibkr_rv"])
        row["MAX_IBKR_RV"] = safe_max(day["ibkr_rv"])
        row["FINAL_IBKR_RV"] = day["ibkr_rv"].iloc[-1]

    if "gp" in day.columns:
        row["MEAN_GP"] = safe_mean(day["gp"])
        row["MAX_ABS_GP"] = day["gp"].abs().max()

    if "macd" in day.columns:
        row["MEAN_MACD"] = safe_mean(day["macd"])
        row["MEAN_ABS_MACD"] = day["macd"].abs().mean()
        row["MAX_ABS_MACD"] = day["macd"].abs().max()

    if "ema9" in day.columns and "ema20" in day.columns:
        spread = (day["ema9"] - day["ema20"]) / day["c"]

        row["MEAN_EMA_SPREAD"] = spread.mean()
        row["MEAN_ABS_EMA_SPREAD"] = spread.abs().mean()
        row["MAX_ABS_EMA_SPREAD"] = spread.abs().max()

    if "f" in day.columns:
        row["FILL_RATE"] = day["f"].mean()
        row["PM_FILL_RATE"] = (
            pm["f"].mean()
            if len(pm) else np.nan
        )

    return row

def add_sample_metrics(row, logits, y, thresholds=np.arange(0.40, 0.701, 0.05)):
    probs = torch.softmax(logits, dim=-1)
    pred = probs.argmax(dim=-1)

    row["CE"] = F.cross_entropy(logits, y).item()
    row["ACC"] = (pred == y).float().mean().item()
    row["MEAN_CONF"] = probs.max(dim=-1).values.mean().item()
    row["MEAN_ENTROPY"] = (-(probs * probs.clamp_min(1e-8).log()).sum(dim=-1).mean().item())

    for cls, name in enumerate(CLASS_NAMES):
        true = y == cls
        predicted = pred == cls
        tp = (true & predicted).sum().item()
        fp = (~true & predicted).sum().item()
        fn = (true & ~predicted).sum().item()

        row[f"{name}_TRUE_N"] = true.sum().item()
        row[f"{name}_PRED_N"] = predicted.sum().item()
        row[f"{name}_PREC"] = tp / (tp + fp) if tp + fp else np.nan
        row[f"{name}_REC"] = tp / (tp + fn) if tp + fn else np.nan
        row[f"{name}_F1"] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan
        row[f"{name}_TRUE_FRAC"] = true.float().mean().item()
        row[f"MEAN_{name}_PROB"] = probs[:, cls].mean().item()
        row[f"MAX_{name}_PROB"] = probs[:, cls].max().item()

    for threshold in thresholds:
        suffix = int(round(threshold * 100))
        for cls, name in [(0, "DOWN"), (2, "UP")]:
            signal = probs[:, cls] >= threshold
            true = y == cls
            tp = (signal & true).sum().item()
            signal_n = signal.sum().item()
            true_n = true.sum().item()

            row[f"{name}{suffix}_N"] = signal_n
            row[f"{name}{suffix}_TP"] = tp
            row[f"{name}{suffix}_PREC"] = tp / signal_n if signal_n else np.nan
            row[f"{name}{suffix}_REC"] = tp / true_n if true_n else np.nan

    return row

@torch.no_grad()
def analyze_ticker_days(
    model,
    dataloader,
    device,
    output_path,
    batch_size=128,
):
    model.eval()

    rows = []
    dataset_idx = 0

    for x, y in dataloader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).long()

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(x)

        logits = logits.float()

        for b in range(x.shape[0]):
            day = get_day_table(
                dataloader.dataset,
                dataset_idx,
            )

            row = describe_day(
                day,
                dataloader.dataset.pm_bars,
            )

            row = add_sample_metrics(
                row,
                logits[b],
                y[b],
            )

            rows.append(row)
            dataset_idx += 1

        print(
            f"\rAnalyzed "
            f"{dataset_idx:,}/"
            f"{len(dataloader.dataset):,} ticker-days",
            end="",
        )

    print()

    df = pd.DataFrame(rows)

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_parquet(
        output_path,
        index=False,
    )

    print(f"Saved: {output_path}")
    print(f"Ticker-days: {len(df):,}")
    print(
        f"Mean CE: "
        f"{df['CE'].mean():.4f}"
    )
    print(
        f"Median CE: "
        f"{df['CE'].median():.4f}"
    )
    print(
        f"Mean ACC: "
        f"{df['ACC'].mean():.4f}"
    )

    return df

def compare_outliers(
    df,
    tail=0.10,
):
    numeric = df.select_dtypes(
        include=np.number
    )

    descriptor_cols = [
        col
        for col in numeric.columns
        if col not in {
            "CE",
            "ACC",
            "MEAN_CONF",
            "MEAN_ENTROPY",

            *[
                f"{name}_{suffix}"
                for name in CLASS_NAMES
                for suffix in [
                    "TRUE_N",
                    "PRED_N",
                    "PREC",
                    "REC",
                    "F1",
                    "TRUE_FRAC",
                ]
            ],

            *[
                f"{prefix}_{name}_PROB"
                for name in CLASS_NAMES
                for prefix in [
                    "MEAN",
                    "MAX",
                ]
            ],

            "DOWN50_N",
            "DOWN50_PREC",
            "DOWN50_REC",

            "UP50_N",
            "UP50_PREC",
            "UP50_REC",
        }
    ]

    low = df["CE"].quantile(tail)
    high = df["CE"].quantile(
        1 - tail
    )

    best = df[
        df["CE"] <= low
    ].copy()

    worst = df[
        df["CE"] >= high
    ].copy()

    comparison = pd.DataFrame({
        "BEST_MEDIAN":
            best[descriptor_cols].median(),

        "WORST_MEDIAN":
            worst[descriptor_cols].median(),

        "ALL_MEDIAN":
            df[descriptor_cols].median(),
    })

    comparison[
        "WORST_MINUS_BEST"
    ] = (
        comparison["WORST_MEDIAN"]
        - comparison["BEST_MEDIAN"]
    )

    comparison[
        "WORST_OVER_BEST"
    ] = (
        comparison["WORST_MEDIAN"]
        / comparison[
            "BEST_MEDIAN"
        ].replace(0, np.nan)
    )

    comparison = comparison.sort_values(
        "WORST_MINUS_BEST",
        key=lambda x: x.abs(),
        ascending=False,
    )

    return best, worst, comparison

def feature_correlations(df):
    exclude = {
        "CE",
        "ACC",

        "DOWN_PREC",
        "DOWN_REC",
        "DOWN_F1",

        "FLAT_PREC",
        "FLAT_REC",
        "FLAT_F1",

        "UP_PREC",
        "UP_REC",
        "UP_F1",

        "DOWN50_PREC",
        "DOWN50_REC",

        "UP50_PREC",
        "UP50_REC",
    }

    numeric = df.select_dtypes(
        include=np.number
    )

    features = [
        col
        for col in numeric.columns

        if col not in exclude

        and not col.endswith(
            "_TRUE_N"
        )

        and not col.endswith(
            "_PRED_N"
        )

        and not col.endswith(
            "_TRUE_FRAC"
        )

        and not col.endswith(
            "_PROB"
        )

        and not col.endswith(
            "50_N"
        )
    ]

    corr = (
        numeric[
            features + ["CE"]
        ]
        .corr(method="spearman")["CE"]
        .drop("CE")
    )

    return corr.sort_values(
        key=lambda x: x.abs(),
        ascending=False,
    )

def regime_analysis(df, feature, bins=5, thresholds=np.arange(0.40, 0.701, 0.05), show_recall=True, show_n=True):
    x = df.dropna(subset=[feature]).copy()
    x["BIN"] = pd.qcut(x[feature], q=bins, duplicates="drop")

    rows = []
    for bin_name, group in x.groupby("BIN", observed=True):
        row = {
            "BIN": bin_name,
            "N": len(group),
            "FEATURE_MEDIAN": group[feature].median(),
            "CE": group["CE"].mean(),
            "ACC": group["ACC"].mean(),
        }

        for threshold in thresholds:
            suffix = int(round(threshold * 100))
            for name in ["UP", "DOWN"]:
                signal_n = group[f"{name}{suffix}_N"].sum()
                tp = group[f"{name}{suffix}_TP"].sum()
                true_n = group[f"{name}_TRUE_N"].sum()

                row[f"{name}{suffix}_PREC"] = tp / signal_n if signal_n else np.nan

                if show_recall:
                    row[f"{name}{suffix}_REC"] = tp / true_n if true_n else np.nan

                if show_n:
                    row[f"{name}{suffix}_N"] = signal_n

        rows.append(row)

    return pd.DataFrame(rows)

def print_extremes(
    df,
    n=20,
):
    cols = [
        "Tk",
        "date",
        "CE",
        "ACC",

        "DOWN_PREC",
        "FLAT_PREC",
        "UP_PREC",

        "UP50_PREC",
        "UP50_REC",
        "UP50_N",

        "RANGE_PCT",
        "TREND_PCT",

        "MAX_IBKR_RV",
        "MEAN_RV",
    ]

    cols = [
        col
        for col in cols
        if col in df.columns
    ]

    print(
        "\nBEST TICKER-DAYS"
    )

    print(
        df
        .nsmallest(n, "CE")[cols]
        .to_string(index=False)
    )

    print(
        "\nWORST TICKER-DAYS"
    )

    print(
        df
        .nlargest(n, "CE")[cols]
        .to_string(index=False)
    )

def run_ticker_day_analysis(
    model,
    dataset,
    device,
    name="test",
    batch_size=128,
    threshold=0.50,
):
    output_folder = Path(
        "analysis/ticker_days"
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = analyze_ticker_days(
        model=model,
        dataset=dataset,
        device=device,
        batch_size=batch_size,
        threshold=threshold,
        output_path=(
            output_folder
            / f"{name}_ticker_days.parquet"
        ),
    )

    print_extremes(df)

    best, worst, comparison = (
        compare_outliers(df)
    )

    comparison.to_parquet(
        output_folder
        / f"{name}_best_vs_worst.parquet"
    )

    corr = feature_correlations(df)

    (
        corr
        .rename("SPEARMAN_CE")
        .to_frame()
        .to_parquet(
            output_folder
            / f"{name}_feature_ce_correlations.parquet"
        )
    )

    best.to_parquet(
        output_folder
        / f"{name}_best_10pct.parquet",
        index=False,
    )

    worst.to_parquet(
        output_folder
        / f"{name}_worst_10pct.parquet",
        index=False,
    )

    print(
        "\nFEATURE CORRELATION WITH CE"
    )

    print(
        corr.head(25)
    )

    print(
        "\nBEST 10% VS WORST 10%"
    )

    print(
        comparison.head(25)
    )

    return (
        df,
        comparison,
        corr,
    )