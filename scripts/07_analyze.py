"""Full analysis: metrics, 5 tables, 7 figures, IoT metrics."""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.metrics import (
    compute_metrics,
    compute_per_item_srs,
    filter_items_by_baseline_srs,
    add_deltas_from_baseline,
    wilson_ci,
)
from src.stats import chi_squared_test, logistic_regression_pruning


# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_all_raw_results(raw_dir):
    """Load all JSONL files from raw_dir into a single DataFrame."""
    pattern = os.path.join(raw_dir, "*.jsonl")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No JSONL files found in {raw_dir}")

    records = []
    for filepath in files:
        with open(filepath) as f:
            for line in f:
                records.append(json.loads(line))

    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} records from {len(files)} files")
    return df


def load_perplexity_results(path):
    """Load perplexity CSV."""
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} perplexity results")
    return df


# ---------------------------------------------------------------------------
# Transition analysis
# ---------------------------------------------------------------------------
def compute_transition_table(df, metrics_df):
    """Track items that are unbiased at baseline but become biased after pruning."""
    letters = ["A", "B", "C"]
    # Get per-item SRS at baseline (dense, sparsity=0)
    baseline_item_srs = compute_per_item_srs(
        df, baseline_filter={"sparsity": 0, "prune_method": "dense"}
    )

    # Unbiased items: SRS == 0 at baseline
    unbiased_items = baseline_item_srs[baseline_item_srs["item_srs"] == 0][
        ["model", "category", "item_id"]
    ].copy()

    # For each sparsity level, compute SRS of these unbiased items
    pruned_df = df[df["sparsity"] > 0].copy()
    pruned_unbiased = pruned_df.merge(unbiased_items, on=["model", "category", "item_id"], how="inner")

    if pruned_unbiased.empty:
        print("  Warning: No unbiased items found for transition analysis")
        return pd.DataFrame(), pd.DataFrame()

    # Per-item SRS at each sparsity
    pruned_unbiased["is_stereo"] = (
        pruned_unbiased["parsed_answer"]
        == pruned_unbiased["stereotype_target_index"].map(lambda i: letters[i])
    )
    pruned_unbiased["is_valid"] = pruned_unbiased["parsed_answer"].notna()

    item_stats = (
        pruned_unbiased
        .groupby(["model", "prune_method", "sparsity", "category", "item_id"])
        .agg(n_valid=("is_valid", "sum"), n_stereo=("is_stereo", "sum"))
        .reset_index()
    )
    item_stats["item_srs"] = np.where(
        item_stats["n_valid"] > 0,
        item_stats["n_stereo"] / item_stats["n_valid"], 0
    )
    # An item "became biased" if SRS > 0
    item_stats["became_biased"] = item_stats["item_srs"] > 0

    # Table 1: Transition summary
    table1 = (
        item_stats
        .groupby(["model", "prune_method", "sparsity"])
        .agg(
            total_unbiased_items=("item_id", "count"),
            became_biased=("became_biased", "sum"),
        )
        .reset_index()
    )
    table1["pct_became_biased"] = (table1["became_biased"] / table1["total_unbiased_items"] * 100).round(2)

    # Table 2: Dose-response (by sparsity level, averaged across methods)
    table2 = (
        table1
        .groupby(["model", "sparsity"])
        .agg(
            avg_pct_became_biased=("pct_became_biased", "mean"),
            total_items=("total_unbiased_items", "mean"),
        )
        .reset_index()
    )
    table2["avg_pct_became_biased"] = table2["avg_pct_became_biased"].round(2)

    return table1, table2


# ---------------------------------------------------------------------------
# Evaluation gap
# ---------------------------------------------------------------------------
def compute_evaluation_gap(metrics_df, ppl_df):
    """Merge metrics with perplexity and compute deltas."""
    # Aggregate metrics across categories for model-level view
    model_metrics = (
        metrics_df
        .groupby(["model", "prune_method", "sparsity"])
        .agg(srs_mean=("srs", "mean"), usr_mean=("usr", "mean"))
        .reset_index()
    )

    merged = model_metrics.merge(ppl_df, on=["model", "prune_method", "sparsity"], how="inner")

    # Compute deltas from dense baseline
    baselines = merged[merged["sparsity"] == 0][["model", "srs_mean", "usr_mean", "perplexity"]].rename(
        columns={"srs_mean": "srs_base", "usr_mean": "usr_base", "perplexity": "ppl_base"}
    )
    merged = merged.merge(baselines, on="model", how="left")
    merged["srs_pct_change"] = ((merged["srs_mean"] - merged["srs_base"]) / merged["srs_base"] * 100).round(2)
    merged["ppl_pct_change"] = ((merged["perplexity"] - merged["ppl_base"]) / merged["ppl_base"] * 100).round(2)

    return merged


# ---------------------------------------------------------------------------
# Method comparison
# ---------------------------------------------------------------------------
def compute_method_comparison(metrics_df):
    """Compare SRS/USR at S50 across pruning methods."""
    s50 = metrics_df[metrics_df["sparsity"] == 50].copy()
    if s50.empty:
        print("  Warning: No S50 data found for method comparison")
        return pd.DataFrame()

    table = (
        s50
        .groupby(["model", "prune_method", "category"])
        .agg(srs=("srs", "mean"), usr=("usr", "mean"))
        .reset_index()
    )
    return table


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------
def run_statistical_tests(df, metrics_df):
    """Chi-squared + Cohen's h for each Dense vs S_x pair, logistic regression per model."""
    results = []

    for model in metrics_df["model"].unique():
        model_metrics = metrics_df[metrics_df["model"] == model]
        baseline = model_metrics[
            (model_metrics["sparsity"] == 0) & (model_metrics["prune_method"] == "dense")
        ]

        for _, base_row in baseline.iterrows():
            category = base_row["category"]
            comparisons = model_metrics[
                (model_metrics["category"] == category)
                & (model_metrics["sparsity"] > 0)
            ]
            for _, comp_row in comparisons.iterrows():
                if base_row["srs"] is None or comp_row["srs"] is None:
                    continue
                test = chi_squared_test(
                    base_row["srs"], base_row["n_valid"],
                    comp_row["srs"], comp_row["n_valid"],
                )
                results.append({
                    "model": model,
                    "category": category,
                    "prune_method": comp_row["prune_method"],
                    "sparsity": comp_row["sparsity"],
                    "chi2": test["chi2"],
                    "p_value": test["p"],
                    "cohens_h": test["cohens_h"],
                    "significant": test["p"] < 0.05 if not np.isnan(test["p"]) else False,
                })

    stat_df = pd.DataFrame(results)

    # Logistic regression per model
    logistic_results = {}
    for model in df["model"].unique():
        model_df = df[df["model"] == model]
        valid = model_df[model_df["parsed_answer"].notna()]
        if len(valid) > 0:
            try:
                summary = logistic_regression_pruning(valid)
                logistic_results[model] = summary
            except Exception as e:
                print(f"  Logistic regression failed for {model}: {e}")

    return stat_df, logistic_results


# ---------------------------------------------------------------------------
# IoT metrics
# ---------------------------------------------------------------------------
def compute_iot_metrics(df, cfg):
    """Parse failure rate, latency, model sizes, response entropy."""
    models_dir = cfg["pruning"]["models_dir"]

    # Parse failure rate per config
    parse_rates = (
        df
        .groupby(["model", "prune_method", "sparsity"])
        .apply(lambda g: pd.Series({
            "total": len(g),
            "parse_failures": g["parsed_answer"].isna().sum(),
            "parse_fail_rate": g["parsed_answer"].isna().mean(),
        }))
        .reset_index()
    )

    # Latency stats
    if "elapsed_s" in df.columns:
        latency = (
            df
            .groupby(["model", "prune_method", "sparsity"])
            .agg(
                latency_mean=("elapsed_s", "mean"),
                latency_median=("elapsed_s", "median"),
                latency_p95=("elapsed_s", lambda x: x.quantile(0.95)),
            )
            .reset_index()
        )
    else:
        latency = pd.DataFrame()

    # Model storage sizes
    sizes = []
    for model_cfg in cfg["models"]:
        name = model_cfg["name"]
        # Dense
        dense_path = os.path.join(models_dir, f"{name}-bf16")
        if os.path.exists(dense_path):
            size = sum(
                os.path.getsize(os.path.join(dense_path, f))
                for f in os.listdir(dense_path)
                if os.path.isfile(os.path.join(dense_path, f))
            )
            sizes.append({"model": name, "prune_method": "dense", "sparsity": 0,
                          "size_bytes": size, "size_gb": round(size / 1e9, 2)})
        # Pruned
        for method in cfg["pruning"]["methods"]:
            for sparsity in cfg["pruning"]["sparsity_levels"]:
                if sparsity == 0:
                    continue
                path = os.path.join(models_dir, f"{name}-{method}-s{sparsity}")
                if os.path.exists(path):
                    size = sum(
                        os.path.getsize(os.path.join(path, f))
                        for f in os.listdir(path)
                        if os.path.isfile(os.path.join(path, f))
                    )
                    sizes.append({"model": name, "prune_method": method, "sparsity": sparsity,
                                  "size_bytes": size, "size_gb": round(size / 1e9, 2)})
    sizes_df = pd.DataFrame(sizes)

    # Response entropy
    valid = df[df["parsed_answer"].notna()].copy()
    entropy_data = (
        valid
        .groupby(["model", "prune_method", "sparsity"])
        .apply(lambda g: pd.Series({
            "response_entropy": -sum(
                (p := g["parsed_answer"].value_counts(normalize=True)) * np.log2(p + 1e-10)
            ),
        }))
        .reset_index()
    )

    return parse_rates, latency, sizes_df, entropy_data


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
MODEL_COLORS = {
    "gemma-2-9b-it": "#1f77b4",
    "mistral-7b-instruct-v0.3": "#ff7f0e",
    "phi-3.5-mini-instruct": "#2ca02c",
}
METHOD_STYLES = {
    "random": ("--", "o"),
    "magnitude": ("-.", "s"),
    "wanda": (":", "D"),
    "dense": ("-", "^"),
}


def fig1_srs_vs_sparsity(metrics_df, figures_dir):
    """SRS vs Sparsity — line plot per model, faceted, colored by method."""
    models = metrics_df["model"].unique()
    methods = [m for m in metrics_df["prune_method"].unique() if m != "dense"]

    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4), sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        model_df = metrics_df[metrics_df["model"] == model]

        for method in methods:
            method_df = model_df[model_df["prune_method"] == method]
            # Include dense baseline at sparsity=0
            dense_df = model_df[model_df["prune_method"] == "dense"]
            combined = pd.concat([dense_df, method_df])
            agg = combined.groupby("sparsity")["srs"].mean().reset_index().sort_values("sparsity")

            ls, marker = METHOD_STYLES.get(method, ("-", "o"))
            ax.plot(agg["sparsity"], agg["srs"], linestyle=ls, marker=marker,
                    label=method, markersize=5)

        ax.set_title(model)
        ax.set_xlabel("Sparsity (%)")
        if ax == axes[0]:
            ax.set_ylabel("SRS")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Stereotype Reliance Score vs Sparsity", y=1.02)
    _save_fig(fig, figures_dir, "fig1_srs_vs_sparsity")


def fig2_transition_bar(table1, figures_dir):
    """Transition bar chart — % of baseline-unbiased items that became biased."""
    if table1.empty:
        print("  Skipping fig2: no transition data")
        return

    models = table1["model"].unique()
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4), sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        model_df = table1[table1["model"] == model]
        methods = model_df["prune_method"].unique()

        bar_width = 0.25
        sparsities = sorted(model_df["sparsity"].unique())
        x = np.arange(len(sparsities))

        for i, method in enumerate(methods):
            method_df2 = model_df[model_df["prune_method"] == method]
            vals = [method_df2[method_df2["sparsity"] == s]["pct_became_biased"].values[0]
                    if s in method_df2["sparsity"].values else 0
                    for s in sparsities]
            ax.bar(x + i * bar_width, vals, bar_width, label=method)

        ax.set_title(model)
        ax.set_xlabel("Sparsity (%)")
        ax.set_xticks(x + bar_width * (len(methods) - 1) / 2)
        ax.set_xticklabels(sparsities)
        if ax == axes[0]:
            ax.set_ylabel("% Became Biased")
        ax.legend(fontsize=8)

    fig.suptitle("Bias Transition: Unbiased → Biased After Pruning", y=1.02)
    _save_fig(fig, figures_dir, "fig2_transition_bar")


def fig3_usr_decline(metrics_df, figures_dir):
    """USR decline — line plot vs sparsity."""
    models = metrics_df["model"].unique()
    methods = [m for m in metrics_df["prune_method"].unique() if m != "dense"]

    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4), sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        model_df = metrics_df[metrics_df["model"] == model]

        for method in methods:
            method_df = model_df[model_df["prune_method"] == method]
            dense_df = model_df[model_df["prune_method"] == "dense"]
            combined = pd.concat([dense_df, method_df])
            agg = combined.groupby("sparsity")["usr"].mean().reset_index().sort_values("sparsity")

            ls, marker = METHOD_STYLES.get(method, ("-", "o"))
            ax.plot(agg["sparsity"], agg["usr"], linestyle=ls, marker=marker,
                    label=method, markersize=5)

        ax.set_title(model)
        ax.set_xlabel("Sparsity (%)")
        if ax == axes[0]:
            ax.set_ylabel("USR")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Unknown Selection Rate vs Sparsity", y=1.02)
    _save_fig(fig, figures_dir, "fig3_usr_decline")


def fig4_evaluation_gap(eval_gap_df, figures_dir):
    """Evaluation gap — dual-axis (perplexity dashed, SRS solid)."""
    if eval_gap_df.empty:
        print("  Skipping fig4: no evaluation gap data")
        return

    models = eval_gap_df["model"].unique()
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4))
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        model_df = eval_gap_df[eval_gap_df["model"] == model]
        # Average across methods
        agg = model_df.groupby("sparsity").agg(
            srs_pct=("srs_pct_change", "mean"),
            ppl_pct=("ppl_pct_change", "mean"),
        ).reset_index().sort_values("sparsity")

        ax.plot(agg["sparsity"], agg["srs_pct"], "o-", color="tab:blue", label="SRS Δ%")
        ax2 = ax.twinx()
        ax2.plot(agg["sparsity"], agg["ppl_pct"], "s--", color="tab:red", label="PPL Δ%")

        ax.set_title(model)
        ax.set_xlabel("Sparsity (%)")
        ax.set_ylabel("SRS Δ%", color="tab:blue")
        ax2.set_ylabel("PPL Δ%", color="tab:red")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")

    fig.suptitle("Evaluation Gap: SRS vs Perplexity Change", y=1.02)
    _save_fig(fig, figures_dir, "fig4_evaluation_gap")


def fig5_method_comparison(method_comp_df, figures_dir):
    """Method comparison — grouped bar at S50."""
    if method_comp_df.empty:
        print("  Skipping fig5: no method comparison data")
        return

    models = method_comp_df["model"].unique()
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4), sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        model_df = method_comp_df[method_comp_df["model"] == model]
        agg = model_df.groupby("prune_method")["srs"].mean().reset_index()

        ax.bar(agg["prune_method"], agg["srs"], color=["#1f77b4", "#ff7f0e", "#2ca02c"])
        ax.set_title(model)
        ax.set_xlabel("Pruning Method")
        if ax == axes[0]:
            ax.set_ylabel("SRS at S50")

    fig.suptitle("Method Comparison: SRS at 50% Sparsity", y=1.02)
    _save_fig(fig, figures_dir, "fig5_method_comparison")


def fig6_category_facets(metrics_df, figures_dir):
    """SRS by bias category — faceted line plots."""
    categories = sorted(metrics_df["category"].unique())
    n_cats = len(categories)
    cols = min(3, n_cats)
    rows = (n_cats + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), sharey=True)
    axes = np.array(axes).flatten()

    for idx, category in enumerate(categories):
        ax = axes[idx]
        cat_df = metrics_df[metrics_df["category"] == category]

        for model in cat_df["model"].unique():
            model_cat = cat_df[cat_df["model"] == model]
            # Average across methods
            agg = model_cat.groupby("sparsity")["srs"].mean().reset_index().sort_values("sparsity")
            color = MODEL_COLORS.get(model, None)
            ax.plot(agg["sparsity"], agg["srs"], "o-", label=model, color=color, markersize=4)

        ax.set_title(category)
        ax.set_xlabel("Sparsity (%)")
        if idx % cols == 0:
            ax.set_ylabel("SRS")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for idx in range(n_cats, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("SRS by Bias Category", y=1.02)
    _save_fig(fig, figures_dir, "fig6_category_facets")


def fig7_latent_bias(metrics_all, metrics_filtered, figures_dir):
    """Latent-bias amplification — all items vs filtered items."""
    if metrics_filtered.empty:
        print("  Skipping fig7: no filtered metrics")
        return

    models = metrics_all["model"].unique()
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4), sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        # All items
        all_model = metrics_all[metrics_all["model"] == model]
        all_agg = all_model.groupby("sparsity")["srs"].mean().reset_index().sort_values("sparsity")
        ax.plot(all_agg["sparsity"], all_agg["srs"], "o-", label="All items", color="tab:blue")

        # Filtered items
        filt_model = metrics_filtered[metrics_filtered["model"] == model]
        filt_agg = filt_model.groupby("sparsity")["srs"].mean().reset_index().sort_values("sparsity")
        ax.plot(filt_agg["sparsity"], filt_agg["srs"], "s--", label="Latent-biased items", color="tab:red")

        ax.set_title(model)
        ax.set_xlabel("Sparsity (%)")
        if ax == axes[0]:
            ax.set_ylabel("SRS")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Latent Bias Amplification: All vs Filtered Items", y=1.02)
    _save_fig(fig, figures_dir, "fig7_latent_bias")


def _save_fig(fig, figures_dir, name):
    """Save figure as both PDF and PNG."""
    fig.savefig(os.path.join(figures_dir, f"{name}.pdf"))
    fig.savefig(os.path.join(figures_dir, f"{name}.png"))
    plt.close(fig)
    print(f"  Saved {name}.pdf and {name}.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def analyze(config_path="config.yaml"):
    cfg = yaml.safe_load(open(config_path))
    raw_dir = cfg["output"]["raw_results_dir"]
    agg_dir = cfg["output"]["aggregated_dir"]
    figures_dir = cfg["output"]["figures_dir"]
    ppl_path = "results/perplexity/perplexity_results.csv"

    for d in [agg_dir, figures_dir, "results/tables", "results/filtered"]:
        os.makedirs(d, exist_ok=True)

    # ----- Load data -----
    print("=" * 60)
    print("Loading data...")
    df = load_all_raw_results(raw_dir)

    # ----- Core metrics -----
    print("\n" + "=" * 60)
    print("Computing core metrics...")
    metrics_df = compute_metrics(df)
    metrics_df = add_deltas_from_baseline(
        metrics_df, baseline_filter={"sparsity": 0, "prune_method": "dense"}
    )
    metrics_df.to_csv(os.path.join(agg_dir, "metrics.csv"), index=False)
    print(f"  Saved metrics.csv ({len(metrics_df)} rows)")

    # ----- Transition analysis -----
    print("\n" + "=" * 60)
    print("Computing transition analysis...")
    table1, table2 = compute_transition_table(df, metrics_df)
    table1.to_csv("results/tables/table1_transition.csv", index=False)
    table2.to_csv("results/tables/table2_dose_response.csv", index=False)
    print(f"  Table 1: {len(table1)} rows, Table 2: {len(table2)} rows")

    # ----- Evaluation gap -----
    eval_gap_df = pd.DataFrame()
    if os.path.exists(ppl_path):
        print("\n" + "=" * 60)
        print("Computing evaluation gap...")
        ppl_df = load_perplexity_results(ppl_path)
        eval_gap_df = compute_evaluation_gap(metrics_df, ppl_df)
        eval_gap_df.to_csv("results/tables/table3_evaluation_gap.csv", index=False)
        print(f"  Table 3: {len(eval_gap_df)} rows")
    else:
        print(f"\nSkipping evaluation gap: {ppl_path} not found")

    # ----- Method comparison -----
    print("\n" + "=" * 60)
    print("Computing method comparison...")
    method_comp = compute_method_comparison(metrics_df)
    method_comp.to_csv("results/tables/table4_method_comparison.csv", index=False)
    print(f"  Table 4: {len(method_comp)} rows")

    # ----- Statistical tests -----
    print("\n" + "=" * 60)
    print("Running statistical tests...")
    stat_df, logistic_results = run_statistical_tests(df, metrics_df)
    stat_df.to_csv("results/tables/table5_statistical_tests.csv", index=False)
    print(f"  Table 5: {len(stat_df)} rows")

    # Save logistic regression summaries
    for model, summary in logistic_results.items():
        summary_path = os.path.join(agg_dir, f"logistic_{model}.txt")
        with open(summary_path, "w") as f:
            f.write(str(summary))
        print(f"  Saved logistic regression for {model}")

    # ----- Latent-bias analysis -----
    print("\n" + "=" * 60)
    print("Computing latent-bias analysis...")
    baseline_filter_cfg = cfg.get("baseline_filter", {"min_srs": 0.20})
    item_srs = compute_per_item_srs(
        df, baseline_filter={"sparsity": 0, "prune_method": "dense"}
    )
    filtered_df, filter_summary = filter_items_by_baseline_srs(
        df, item_srs,
        min_srs=baseline_filter_cfg.get("min_srs", 0.20),
        max_srs=baseline_filter_cfg.get("max_srs", 1.01),
    )
    filter_summary.to_csv("results/filtered/filter_summary.csv", index=False)
    print(f"  Filter summary: {len(filter_summary)} rows")

    metrics_filtered = pd.DataFrame()
    if not filtered_df.empty:
        metrics_filtered = compute_metrics(filtered_df)
        metrics_filtered = add_deltas_from_baseline(
            metrics_filtered, baseline_filter={"sparsity": 0, "prune_method": "dense"}
        )
        metrics_filtered.to_csv("results/filtered/metrics_filtered.csv", index=False)
        print(f"  Filtered metrics: {len(metrics_filtered)} rows")

    # ----- IoT metrics -----
    print("\n" + "=" * 60)
    print("Computing IoT metrics...")
    parse_rates, latency, sizes_df, entropy_data = compute_iot_metrics(df, cfg)
    parse_rates.to_csv(os.path.join(agg_dir, "parse_failure_rates.csv"), index=False)
    if not latency.empty:
        latency.to_csv(os.path.join(agg_dir, "latency_stats.csv"), index=False)
    if not sizes_df.empty:
        sizes_df.to_csv(os.path.join(agg_dir, "model_sizes.csv"), index=False)
    entropy_data.to_csv(os.path.join(agg_dir, "response_entropy.csv"), index=False)
    print("  Saved IoT metric CSVs")

    # ----- Figures -----
    print("\n" + "=" * 60)
    print("Generating figures...")

    fig1_srs_vs_sparsity(metrics_df, figures_dir)
    fig2_transition_bar(table1, figures_dir)
    fig3_usr_decline(metrics_df, figures_dir)
    fig4_evaluation_gap(eval_gap_df, figures_dir)
    fig5_method_comparison(method_comp, figures_dir)
    fig6_category_facets(metrics_df, figures_dir)
    fig7_latent_bias(metrics_df, metrics_filtered, figures_dir)

    print("\n" + "=" * 60)
    print("Analysis complete. Outputs:")
    print(f"  Tables:     results/tables/")
    print(f"  Figures:    {figures_dir}/")
    print(f"  Aggregated: {agg_dir}/")
    print(f"  Filtered:   results/filtered/")


if __name__ == "__main__":
    analyze()
