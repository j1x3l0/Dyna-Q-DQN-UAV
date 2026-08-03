"""Validate and compare the formal Stage-D K=1/K=5 paired ablation."""

import argparse
import csv
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRICS = (
    "final_reward", "final_xi_50", "early_reward_500", "auc_reward_500",
    "auc_reward_1000", "convergence_episode", "episodes_completed",
    "duration_hours", "final_model_loss_50", "final_delivered_50",
    "final_received_50", "final_total_energy_50", "final_collisions_50",
    "collision_events_per_episode", "final_forward_receive_ratio_50",
)


def exact_sign_flip(diff):
    values = np.asarray(diff, dtype=float)
    observed = float(values.mean())
    null = np.asarray([
        np.mean(values * signs)
        for signs in itertools.product((-1, 1), repeat=len(values))
    ])
    tol = 1e-12
    return {
        "mean_difference_k5_minus_k1": observed,
        "p_exact_two_sided": float(np.mean(np.abs(null) >= abs(observed) - tol)),
        "p_exact_greater": float(np.mean(null >= observed - tol)),
        "p_exact_less": float(np.mean(null <= observed + tol)),
    }


def exact_bootstrap_ci(diff):
    values = np.asarray(diff, dtype=float)
    means = np.asarray([
        np.mean(values[list(indices)])
        for indices in itertools.product(range(len(values)), repeat=len(values))
    ])
    return [float(x) for x in np.percentile(means, [2.5, 97.5])]


def holm(p_values):
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def tail_mean(rows, key, window=50):
    return float(np.mean([float(row[key]) for row in rows[-window:]]))


def run_metrics(run):
    rewards = np.asarray(run["rewards"], dtype=float)
    rows = run["episode_metrics"]
    delivered = tail_mean(rows, "data_sent_to_rbs")
    received = tail_mean(rows, "data_received")
    return {
        "variant": run["variant"],
        "seed": int(run["seed"]),
        "episodes_completed": int(run["episodes_completed"]),
        "stopped_early": bool(run["stopped_early"]),
        "final_reward": float(np.mean(rewards[-50:])),
        "final_xi_50": tail_mean(rows, "paper_xi_ratio_of_sums"),
        "early_reward_500": float(np.mean(rewards[:500])),
        "auc_reward_500": float(np.trapezoid(rewards[:500]) / 499),
        "auc_reward_1000": float(np.trapezoid(rewards[:1000]) / 999),
        "convergence_episode": float(run["convergence_episode"]),
        "duration_hours": float(run["duration"] / 3600),
        "final_model_loss_50": float(run["final_model_loss_50"]),
        "final_delivered_50": delivered,
        "final_received_50": received,
        "final_total_energy_50": tail_mean(rows, "flight_energy")
        + tail_mean(rows, "energy_consumed"),
        "final_collisions_50": tail_mean(rows, "collision_events"),
        "collision_events_per_episode": float(
            sum(row["collision_events"] for row in rows) / len(rows)),
        "final_forward_receive_ratio_50": float(delivered / received),
    }


def summarize(values):
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()), "std": float(arr.std(ddof=1)),
        "median": float(np.median(arr)),
        "ci95_normal": float(1.96 * arr.std(ddof=1) / np.sqrt(len(arr))),
    }


def plot_results(raw_runs, paired, output_dir):
    colors = {"k1_w32": "#2563EB", "k5_w32": "#D97706"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, metric, label in zip(
        axes.flat[:3], ("final_xi_50", "convergence_episode", "duration_hours"),
        ("Final Xi (last 50)", "Convergence episode (lower is better)",
         "Wall-clock hours"),
    ):
        for seed in sorted(paired):
            vals = [paired[seed][v][metric] for v in ("k1_w32", "k5_w32")]
            ax.plot([1, 5], vals, color="#9CA3AF", marker="o", linewidth=1)
        means = [np.mean([paired[s][v][metric] for s in paired])
                 for v in ("k1_w32", "k5_w32")]
        ax.plot([1, 5], means, color="#111827", marker="D", linewidth=2.5,
                label="Mean")
        ax.set(xticks=[1, 5], xlabel="Dyna planning steps K", ylabel=label)
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax = axes.flat[3]
    horizon, window = 2500, 100
    for variant in ("k1_w32", "k5_w32"):
        curves = []
        for run in raw_runs:
            if run["variant"] == variant:
                rewards = np.asarray(run["rewards"][:horizon], dtype=float)
                curves.append(np.convolve(rewards, np.ones(window) / window, "valid"))
        curves = np.vstack(curves)
        x = np.arange(window, horizon + 1)
        mean, ci = curves.mean(0), 1.96 * curves.std(0, ddof=1) / np.sqrt(len(curves))
        ax.plot(x, mean, color=colors[variant], label=variant.replace("_w32", ""))
        ax.fill_between(x, mean - ci, mean + ci, color=colors[variant], alpha=0.18)
    ax.set(xlabel="Episode", ylabel="Reward (100-episode rolling mean)")
    ax.grid(color="#E5E7EB", linewidth=0.8)
    ax.legend(frameon=False)
    fig.suptitle("Stage-D paired Dyna-K ablation (5 seeds, warm-up=32)")
    fig.tight_layout()
    fig.savefig(output_dir / "stage_d_k_ablation_main.png", dpi=220)
    plt.close(fig)

    labels = ["Delivered", "Received", "Total energy", "Model loss"]
    keys = ["final_delivered_50", "final_received_50", "final_total_energy_50",
            "final_model_loss_50"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    for ax, key, label in zip(axes, keys, labels):
        means, errors = [], []
        for variant in ("k1_w32", "k5_w32"):
            vals = [row[key] for row in paired.values() for name, row in row.items()
                    if name == variant]
            means.append(np.mean(vals)); errors.append(1.96 * np.std(vals, ddof=1) / np.sqrt(5))
        ax.bar(["K=1", "K=5"], means, yerr=errors,
               color=[colors["k1_w32"], colors["k5_w32"]], capsize=4)
        ax.set_title(label); ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    fig.suptitle("Final-window system metrics (mean and normal 95% CI, n=5)")
    fig.tight_layout()
    fig.savefig(output_dir / "stage_d_k_ablation_system_metrics.png", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    source = json.loads(args.input_json.read_text())
    if len(source.get("runs", [])) != 10:
        raise ValueError("expected exactly 10 formal runs")
    metrics = [run_metrics(run) for run in source["runs"]]
    paired = {}
    for row in metrics:
        paired.setdefault(row["seed"], {})[row["variant"]] = row
    if sorted(paired) != [42, 123, 2026, 3407, 8888] or any(len(v) != 2 for v in paired.values()):
        raise ValueError("expected five complete paired seeds")
    if any(len(run["rewards"]) != run["episodes_completed"]
           or len(run["episode_metrics"]) != run["episodes_completed"] for run in source["runs"]):
        raise ValueError("episode array length mismatch")
    if not all(np.isfinite([x for row in metrics for key, x in row.items()
                            if key not in ("variant", "stopped_early")])):
        raise ValueError("non-finite metric detected")

    tests, summaries = {}, {}
    for metric in METRICS:
        k1 = [paired[s]["k1_w32"][metric] for s in sorted(paired)]
        k5 = [paired[s]["k5_w32"][metric] for s in sorted(paired)]
        diff = np.asarray(k5) - np.asarray(k1)
        tests[metric] = exact_sign_flip(diff)
        tests[metric]["bootstrap_ci95"] = exact_bootstrap_ci(diff)
        tests[metric]["paired_differences"] = diff.tolist()
        tests[metric]["cohens_dz"] = float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) else None
        summaries[metric] = {"k1": summarize(k1), "k5": summarize(k5)}
    primary = ["final_xi_50", "convergence_episode", "duration_hours"]
    adjusted = holm([tests[m]["p_exact_two_sided"] for m in primary])
    for metric, value in zip(primary, adjusted):
        tests[metric]["p_holm_two_sided_primary"] = value

    args.output_dir.mkdir(parents=True, exist_ok=True)
    compact = {"source_sha256": "66152151c7b7fb25904fcc0e973820a0f750095a046bda4ad7fbcf8d2c754c04",
               "settings": source["settings"], "seeds": sorted(paired),
               "validation": {"runs": 10, "all_early_stopped": all(r["stopped_early"] for r in metrics),
                              "array_lengths_match": True, "finite_metrics": True},
               "per_seed": metrics, "summaries": summaries, "paired_tests": tests}
    (args.output_dir / "stage_d_k_ablation_analysis_20260803.json").write_text(
        json.dumps(compact, ensure_ascii=False, indent=2))
    with (args.output_dir / "stage_d_k_ablation_per_seed_20260803.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics[0].keys()); writer.writeheader(); writer.writerows(metrics)
    plot_results(source["runs"], paired, args.output_dir)
    print(args.output_dir)


if __name__ == "__main__":
    main()
