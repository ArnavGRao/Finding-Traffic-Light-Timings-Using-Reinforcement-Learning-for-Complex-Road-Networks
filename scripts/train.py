"""Unified training CLI for baseline and D3QN ablation traffic signal models."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.contracts import EPISODE_SUMMARY_COLUMNS, load_default_config
from src.utils.experiment import (
    DEMAND_CHOICES,
    MODEL_CHOICES,
    MODEL_CONFIGS,
    NETWORK_CHOICES,
    append_csv_row,
    build_controller_or_agent,
    build_eval_row,
    build_experiment_env,
    load_agent_checkpoint,
    run_episode,
    save_agent_checkpoint,
)

TRAINING_CURVE_COLUMNS = [
    "episode",
    "model",
    "network",
    "demand",
    "seed",
    "global_step",
    "epsilon",
    "total_reward",
    "mean_loss",
    "avg_vehicle_delay_s_per_veh",
    "max_lane_wait_s",
    "lane_wait_variance_s2",
    "max_queue_vehicles",
    "throughput_veh_per_hr",
    "total_co2_g_per_km",
    "oscillation_switches_per_hr",
    "blocked_switches",
    "wall_time_s",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train traffic signal control models and save per-episode artifacts.")
    parser.add_argument("--model", type=str, default="d3qn_proposed", choices=MODEL_CHOICES)
    parser.add_argument("--network", type=str, default="single_intersection", choices=NETWORK_CHOICES)
    parser.add_argument("--demand", type=str, default="balanced", choices=DEMAND_CHOICES)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="outputs/runs")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional .pt checkpoint to resume training from.")
    parser.add_argument("--resume", action="store_true", help="Resume from <output-dir>/<run>/latest.pt if present.")
    parser.add_argument("--checkpoint-every", type=int, default=1, help="Save periodic ep_XXXX.pt every N episodes.")
    parser.add_argument("--config", type=str, default=str(REPO_ROOT / "configs/default.json"))
    parser.add_argument("--route-file", type=str, default=None, help="Optional custom SUMO .rou.xml override.")
    parser.add_argument("--num-seconds", type=int, default=3600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-capacity", type=int, default=100_000)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay-steps", type=int, default=25_000)
    parser.add_argument("--gui", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = load_default_config(args.config)
    model_cfg = MODEL_CONFIGS[args.model]
    is_rl = model_cfg.kind in ("vanilla_dqn", "d3qn")

    run_dir = Path(args.output_dir) / f"{args.model}_{args.network}_{args.demand}"
    ckpt_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    if is_rl:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    latest_ckpt = run_dir / "latest.pt"
    best_ckpt = run_dir / "best.pt"
    curve_csv = run_dir / "training_curve.csv"
    eval_rows_csv = run_dir / "per_seed_eval.csv"

    env = build_experiment_env(
        cfg=cfg,
        model_cfg=model_cfg,
        network=args.network,
        demand=args.demand,
        num_seconds=args.num_seconds,
        route_file_override=args.route_file,
        use_gui=args.gui,
        validate_contracts=False,
    )

    try:
        num_actions = env.num_actions_map[env.ts_id]
        controller = build_controller_or_agent(
            model_cfg,
            num_actions=num_actions,
            cfg=cfg,
            lr=args.lr,
            gamma=args.gamma,
            batch_size=args.batch_size,
            buffer_capacity=args.buffer_capacity,
            eps_start=args.eps_start,
            eps_end=args.eps_end,
            eps_decay_steps=args.eps_decay_steps,
        )

        start_ep = 0
        global_step = 0
        best_reward = -float("inf")

        resume_path: Path | None = None
        if args.checkpoint is not None:
            resume_path = Path(args.checkpoint)
        elif args.resume and latest_ckpt.exists():
            resume_path = latest_ckpt

        if resume_path is not None and is_rl:
            meta = load_agent_checkpoint(resume_path, controller, model_cfg)
            start_ep = max(0, meta["episode"] + 1)
            global_step = meta["global_step"]
            best_reward = meta["best_reward"]
            print(
                f"Resumed {args.model} ({args.network}/{args.demand}) from {resume_path} "
                f"at episode={start_ep}, global_step={global_step}"
            )

        for ep in range(start_ep, args.episodes):
            t0 = time.time()
            ep_seed = args.seed + ep
            result = run_episode(
                env=env,
                controller=controller,
                model_cfg=model_cfg,
                seed=ep_seed,
                train=is_rl,
                global_step=global_step,
                warmup_steps=args.warmup_steps,
                eps_start=args.eps_start,
                eps_end=args.eps_end,
                eps_decay_steps=args.eps_decay_steps,
                episode_idx=ep + 1,
            )
            wall_s = time.time() - t0
            global_step = result["global_step"]
            ep_reward = result["total_reward"]
            summary = result["metrics_summary"]

            if is_rl:
                if ep_reward >= best_reward:
                    best_reward = ep_reward
                    save_agent_checkpoint(
                        best_ckpt,
                        controller,
                        model_cfg,
                        args.network,
                        args.demand,
                        ep,
                        global_step,
                        best_reward,
                    )
                save_agent_checkpoint(
                    latest_ckpt,
                    controller,
                    model_cfg,
                    args.network,
                    args.demand,
                    ep,
                    global_step,
                    best_reward,
                )
                if args.checkpoint_every > 0 and ((ep + 1) % args.checkpoint_every == 0 or (ep + 1) == args.episodes):
                    periodic_ckpt = ckpt_dir / f"ep_{ep + 1:04d}.pt"
                    save_agent_checkpoint(
                        periodic_ckpt,
                        controller,
                        model_cfg,
                        args.network,
                        args.demand,
                        ep,
                        global_step,
                        best_reward,
                    )

            curve_row = {
                "episode": ep + 1,
                "model": args.model,
                "network": args.network,
                "demand": args.demand,
                "seed": ep_seed,
                "global_step": global_step,
                "epsilon": round(result["epsilon"], 5),
                "total_reward": round(ep_reward, 5),
                "mean_loss": round(result["mean_loss"], 6),
                "avg_vehicle_delay_s_per_veh": round(float(summary["avg_vehicle_delay_s_per_veh"]), 4),
                "max_lane_wait_s": round(float(summary["max_lane_wait_s"]), 4),
                "lane_wait_variance_s2": round(float(summary["lane_wait_variance_s2"]), 4),
                "max_queue_vehicles": round(float(summary["max_queue_vehicles"]), 4),
                "throughput_veh_per_hr": round(float(summary["throughput_veh_per_hr"]), 4),
                "total_co2_g_per_km": round(float(summary["total_co2_g_per_km"]), 4),
                "oscillation_switches_per_hr": round(float(summary["oscillation_switches_per_hr"]), 4),
                "blocked_switches": result["blocked_switches"],
                "wall_time_s": round(wall_s, 2),
            }
            append_csv_row(curve_csv, TRAINING_CURVE_COLUMNS, curve_row)

            eval_row = build_eval_row(
                summary,
                model_name=model_cfg.display_name,
                network=args.network,
                demand_regime=args.demand,
                seed=ep_seed,
            )
            append_csv_row(eval_rows_csv, EPISODE_SUMMARY_COLUMNS, eval_row)

            print(
                f"[Ep {ep + 1:03d}/{args.episodes:03d}] {args.model} ({args.network}/{args.demand}) | "
                f"R={ep_reward:8.2f} | Loss={result['mean_loss']:.4f} | "
                f"Delay={summary['avg_vehicle_delay_s_per_veh']:.2f}s | "
                f"MaxWait={summary['max_lane_wait_s']:.1f}s | "
                f"Thru={summary['throughput_veh_per_hr']:.1f}/hr | ({wall_s:.1f}s)"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
