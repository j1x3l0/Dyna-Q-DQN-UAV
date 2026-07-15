#!/usr/bin/env python3
"""Run reproducible Dyna-k experiments across multiple random seeds."""

import argparse
import csv
import gc
import json
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hierarchical_agent import HierarchicalAgent  # noqa: E402
from system_model import Config, Environment  # noqa: E402

logging.getLogger("system_model").setLevel(logging.ERROR)
logging.getLogger("hierarchical_agent").setLevel(logging.ERROR)


EPISODE_METRICS = (
    "collision_events",
    "collision_penalty",
    "data_collected",
    "data_delivered",
    "energy_consumed",
    "energy_harvested",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run reproducible Dyna-Q k experiments")
    parser.add_argument("--k-values", type=int, nargs="+", default=[0, 1, 3])
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=None, help="Single-seed compatibility option")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--case", type=int, choices=(1, 2), default=1)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--model-eval-every", type=int, default=10)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "results" / "k_experiments")
    args = parser.parse_args()

    if args.seed is not None and args.seeds is not None:
        parser.error("Use either --seed or --seeds, not both")
    args.seeds = [args.seed] if args.seed is not None else (args.seeds or [42, 123, 2026])
    args.k_values = list(dict.fromkeys(args.k_values))
    args.seeds = list(dict.fromkeys(args.seeds))

    if any(k < 0 for k in args.k_values):
        parser.error("k must be non-negative")
    if args.episodes <= 0:
        parser.error("episodes must be positive")
    if args.warmup_steps < 0:
        parser.error("warmup-steps must be non-negative")
    return args


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def derive_stream_seeds(seed):
    environment_sequence, agent_sequence = np.random.SeedSequence(seed).spawn(2)
    return (
        int(environment_sequence.generate_state(1, dtype=np.uint32)[0]),
        int(agent_sequence.generate_state(1, dtype=np.uint32)[0]),
    )


def make_logger(log_path):
    logger = logging.getLogger(f"k_experiment.{log_path.parent.parent.name}.{log_path.parent.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    for handler in (logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def moving_average(values, window):
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        return values.copy()
    return np.convolve(values, np.ones(window) / window, mode="valid")


@torch.no_grad()
def evaluate_model_error(agent, max_samples=128):
    reward_errors = []
    state_errors = []
    for agent_idx, memory in enumerate(agent.lower_memory):
        if not memory:
            continue
        sample_count = min(len(memory), max_samples)
        indices = np.linspace(0, len(memory) - 1, sample_count, dtype=int)
        samples = [memory[idx] for idx in indices]
        states = torch.as_tensor(np.stack([item[0] for item in samples]), dtype=torch.float32, device=agent.device)
        actions = torch.as_tensor(np.stack([item[1] for item in samples]), dtype=torch.float32, device=agent.device)
        rewards = torch.as_tensor([item[2] for item in samples], dtype=torch.float32, device=agent.device)
        next_states = torch.as_tensor(np.stack([item[3] for item in samples]), dtype=torch.float32, device=agent.device)
        model = agent.models[agent_idx]
        was_training = model.training
        model.eval()
        reward_pred, next_state_pred = model(states, actions)
        model.train(was_training)
        reward_errors.append(torch.mean((reward_pred.squeeze(1) - rewards) ** 2).item())
        state_errors.append(torch.mean((next_state_pred - next_states) ** 2).item())
    if not reward_errors:
        return None, None
    return float(np.mean(reward_errors)), float(np.mean(state_errors))


def agent_state(agent):
    return {
        "upper_actors": [network.state_dict() for network in agent.upper_actors],
        "upper_critics": [network.state_dict() for network in agent.upper_critics],
        "target_upper_actors": [network.state_dict() for network in agent.target_upper_actors],
        "target_upper_critics": [network.state_dict() for network in agent.target_upper_critics],
        "lower_dqns": [network.state_dict() for network in agent.lower_dqns],
        "target_lower_dqns": [network.state_dict() for network in agent.target_lower_dqns],
        "models": [network.state_dict() for network in agent.models],
        "epsilon": agent.epsilon,
    }


def save_checkpoint(path, agent, metadata, rewards):
    torch.save(
        {
            **metadata,
            "rewards": rewards,
            "agent": agent_state(agent),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
        path,
    )


def write_csv(path, rows, fieldnames=None):
    if not rows:
        return
    fieldnames = fieldnames or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_run_plot(path, rows, k, seed):
    episodes = np.asarray([row["episode"] for row in rows])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    series = (
        ("total_reward", "Total reward"),
        ("collision_events", "Collision penalty events"),
        ("data_delivered", "Data delivered to RBS"),
        ("energy_consumed", "Energy consumed"),
    )
    for axis, (key, label) in zip(axes.flat, series):
        values = np.asarray([row[key] for row in rows], dtype=float)
        axis.plot(episodes, values, alpha=0.2)
        smooth = moving_average(values, 10)
        start = len(episodes) - len(smooth)
        axis.plot(episodes[start:], smooth, linewidth=2)
        axis.set_title(label)
        axis.set_xlabel("Episode")
        axis.grid(alpha=0.25)
    fig.suptitle(f"Dyna-Q diagnostics (k={k}, seed={seed})")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def mean_last(values, count):
    return float(np.mean(np.asarray(values, dtype=float)[-count:]))


def train_one(k, seed, args, group_dir):
    set_global_seed(seed)
    environment_seed, agent_seed = derive_stream_seeds(seed)
    run_dir = group_dir / f"k{k}" / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    logger = make_logger(run_dir / "training.log")
    env_config = Config()

    config_data = {
        "k": k,
        "episodes": args.episodes,
        "case": args.case,
        "seed": seed,
        "environment_seed": environment_seed,
        "agent_seed": agent_seed,
        "warmup_steps": args.warmup_steps,
        "checkpoint_every": args.checkpoint_every,
        "model_eval_every": args.model_eval_every,
        "collision_penalty_eta": env_config.eta,
        "energy_penalty_eta1": env_config.eta1,
        "minimum_collision_distance": env_config.d_min,
        "severe_failure_reward_threshold": -50.0 * env_config.eta,
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    (run_dir / "config.json").write_text(json.dumps(config_data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        "Starting k=%d case=%d episodes=%d seed=%d warmup=%d device=%s",
        k,
        args.case,
        args.episodes,
        seed,
        args.warmup_steps,
        config_data["device"],
    )

    env = Environment(env_config, seed=environment_seed)
    agent = HierarchicalAgent(
        30,
        4 + 2 * env_config.M + 1,
        env_config.N,
        env_config,
        seed=agent_seed,
        dyna_k=k,
    )

    rows = []
    rewards_history = []
    global_step = 0
    start_time = time.perf_counter()
    last_reward_mse = None
    last_state_mse = None

    for episode in range(1, args.episodes + 1):
        states = env.reset(args.case)
        total_reward = 0.0
        episode_metrics = {key: 0.0 for key in EPISODE_METRICS}

        while True:
            upper_actions = agent.upper_act(states)
            lower_actions = agent.lower_act(states)
            actions = np.asarray([
                np.concatenate([upper_actions[idx], lower_actions[idx]])
                for idx in range(env_config.N)
            ])
            next_states, rewards, done = env.step(actions)
            for key in EPISODE_METRICS:
                episode_metrics[key] += env.last_step_metrics[key]

            agent.add_upper_memory(states, upper_actions, rewards, next_states, done)
            agent.update_upper()
            for agent_idx in range(env_config.N):
                agent.add_lower_memory(
                    agent_idx,
                    states[agent_idx],
                    lower_actions[agent_idx],
                    rewards[agent_idx],
                    next_states[agent_idx],
                    done,
                )
                agent.update_lower(agent_idx)
                if k > 0:
                    agent.update_model(agent_idx)
                    if global_step >= args.warmup_steps:
                        agent.dyna_plan(agent_idx, k=k)

            total_reward += float(np.sum(rewards))
            states = next_states
            global_step += 1
            if done:
                break

        rewards_history.append(total_reward)
        if k > 0 and (episode % args.model_eval_every == 0 or episode == args.episodes):
            last_reward_mse, last_state_mse = evaluate_model_error(agent)
        agent.end_episode(step_model_scheduler=k > 0)
        elapsed = time.perf_counter() - start_time
        row = {
            "episode": episode,
            "total_reward": total_reward,
            "moving_avg_10": mean_last(rewards_history, 10),
            "moving_avg_50": mean_last(rewards_history, 50),
            "epsilon": agent.epsilon,
            "global_step": global_step,
            "elapsed_seconds": elapsed,
            "model_reward_mse": "" if last_reward_mse is None else last_reward_mse,
            "model_state_mse": "" if last_state_mse is None else last_state_mse,
            "actor_lr": agent.upper_actor_optimizers[0].param_groups[0]["lr"],
            "lower_lr": agent.lower_optimizers[0].param_groups[0]["lr"],
                "model_lr": (
                    agent.model_optimizers[0].param_groups[0]["lr"]
                    if agent.model_optimizers
                    else ""
                ),
            **episode_metrics,
        }
        rows.append(row)

        if episode == 1 or episode % 10 == 0 or episode == args.episodes:
            logger.info(
                "episode=%d/%d reward=%.2f avg50=%.2f collisions=%.0f delivered=%.3f energy=%.3f epsilon=%.4f elapsed=%.1fs",
                episode,
                args.episodes,
                total_reward,
                row["moving_avg_50"],
                row["collision_events"],
                row["data_delivered"],
                row["energy_consumed"],
                agent.epsilon,
                elapsed,
            )

        if episode % args.checkpoint_every == 0 or episode == args.episodes:
            write_csv(run_dir / "metrics.csv", rows)
            np.save(run_dir / "rewards.npy", np.asarray(rewards_history, dtype=float))
            save_checkpoint(
                run_dir / "checkpoint_latest.pt",
                agent,
                {"k": k, "case": args.case, "seed": seed, "episode": episode, "global_step": global_step},
                rewards_history,
            )

    elapsed = time.perf_counter() - start_time
    rewards = np.asarray(rewards_history, dtype=float)
    summary = {
        **config_data,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": elapsed,
        "episodes_per_second": args.episodes / elapsed,
        "environment_steps": global_step,
        "planning_samples": max(global_step - args.warmup_steps, 0) * env_config.N * k,
        "mean_reward": float(np.mean(rewards)),
        "final_avg_10": mean_last(rewards, 10),
        "final_avg_50": mean_last(rewards, 50),
        "best_reward": float(np.max(rewards)),
        "worst_reward": float(np.min(rewards)),
        "reward_std": float(np.std(rewards)),
        "severe_failure_episodes": int(
            np.sum(rewards <= config_data["severe_failure_reward_threshold"])
        ),
        "final_model_reward_mse": last_reward_mse,
        "final_model_state_mse": last_state_mse,
    }
    for key in EPISODE_METRICS:
        values = [row[key] for row in rows]
        summary[f"total_{key}"] = float(np.sum(values))
        summary[f"mean_{key}"] = float(np.mean(values))
        summary[f"final50_{key}"] = mean_last(values, 50)
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    save_run_plot(run_dir / "training_diagnostics.png", rows, k, seed)
    logger.info("Completed k=%d seed=%d in %.1fs final_avg_50=%.2f", k, seed, elapsed, summary["final_avg_50"])
    return {"summary": summary, "rows": rows, "rewards": rewards}


def aggregate_results(group_dir, results_by_k):
    all_summaries = [run["summary"] for runs in results_by_k.values() for run in runs]
    summary_fields = [
        "k", "seed", "final_avg_10", "final_avg_50", "mean_reward", "reward_std",
        "severe_failure_episodes", "elapsed_seconds", "planning_samples",
        "mean_collision_events", "mean_data_collected", "mean_data_delivered",
        "mean_energy_consumed", "mean_energy_harvested",
        "final_model_reward_mse", "final_model_state_mse",
    ]
    write_csv(group_dir / "all_runs.csv", [{key: item.get(key, "") for key in summary_fields} for item in all_summaries], summary_fields)

    aggregate_rows = []
    aggregate_metrics = (
        "final_avg_50", "mean_reward", "reward_std", "severe_failure_episodes",
        "elapsed_seconds", "mean_collision_events", "mean_data_collected", "mean_data_delivered",
        "mean_energy_consumed", "mean_energy_harvested",
    )
    for k, runs in results_by_k.items():
        row = {"k": k, "n_seeds": len(runs)}
        for metric in aggregate_metrics:
            values = np.asarray([run["summary"][metric] for run in runs], dtype=float)
            row[f"{metric}_mean"] = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95"] = 1.96 * std / np.sqrt(len(values)) if len(values) > 1 else 0.0
        aggregate_rows.append(row)
    write_csv(group_dir / "aggregate_by_k.csv", aggregate_rows)

    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    plot_specs = (
        ("total_reward", "Reward (moving average 10)"),
        ("collision_events", "Collision events (moving average 10)"),
        ("data_delivered", "Data delivered (moving average 10)"),
        ("energy_consumed", "Energy consumed (moving average 10)"),
    )
    for axis, (metric, title) in zip(axes.flat, plot_specs):
        for k, runs in results_by_k.items():
            sequences = []
            for run in runs:
                values = run["rewards"] if metric == "total_reward" else [row[metric] for row in run["rows"]]
                sequences.append(moving_average(values, 10))
            stacked = np.stack(sequences)
            mean = np.mean(stacked, axis=0)
            std = np.std(stacked, axis=0)
            episode_count = len(runs[0]["rows"])
            x = np.arange(10, 10 + len(mean)) if episode_count >= 10 else np.arange(1, 1 + len(mean))
            axis.plot(x, mean, linewidth=2, label=f"k={k}")
            axis.fill_between(x, mean - std, mean + std, alpha=0.15)
        axis.set_title(title)
        axis.set_xlabel("Episode")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Dyna-Q k comparison across seeds (mean ± standard deviation)")
    figure.tight_layout()
    figure.savefig(group_dir / "comparison_diagnostics.png", dpi=160)
    plt.close(figure)

    best = max(aggregate_rows, key=lambda row: (row["final_avg_50_mean"], -row["k"]))
    comparison = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "best_k_by_mean_final_avg_50": int(best["k"]),
        "aggregate": aggregate_rows,
        "runs": all_summaries,
    }
    (group_dir / "comparison_summary.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    k_label = "-".join(str(k) for k in args.k_values)
    seed_label = "-".join(str(seed) for seed in args.seeds)
    group_dir = args.output_root / f"case{args.case}_ep{args.episodes}_k{k_label}_seeds{seed_label}_{timestamp}"
    group_dir.mkdir(parents=True, exist_ok=False)
    experiment_config = vars(args).copy()
    experiment_config["output_root"] = str(experiment_config["output_root"])
    base_env_config = Config()
    experiment_config.update(
        {
            "collision_penalty_eta": base_env_config.eta,
            "energy_penalty_eta1": base_env_config.eta1,
            "minimum_collision_distance": base_env_config.d_min,
            "severe_failure_reward_threshold": -50.0 * base_env_config.eta,
        }
    )
    (group_dir / "experiment_config.json").write_text(
        json.dumps(experiment_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    results_by_k = {k: [] for k in args.k_values}
    for k in args.k_values:
        for seed in args.seeds:
            results_by_k[k].append(train_one(k, seed, args, group_dir))
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    aggregate_results(group_dir, results_by_k)
    print(f"RESULT_DIR={group_dir}", flush=True)


if __name__ == "__main__":
    main()
