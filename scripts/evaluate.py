"""Unified deterministic evaluation CLI for baseline and D3QN traffic signal models."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.contracts import EPISODE_SUMMARY_COLUMNS, load_default_config
from src.utils.demand_generator import generate_random_demand_route
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
    write_summary_aggregates_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baseline and RL traffic models across seeds.")
    parser.add_argument("--model", type=str, required=True, choices=MODEL_CHOICES)
    parser.add_argument("--network", type=str, default="single_intersection", choices=NETWORK_CHOICES)
    parser.add_argument("--demand", type=str, default="balanced", choices=[*DEMAND_CHOICES, "random_stress"])
    parser.add_argument("--episodes", type=int, default=50, help="Number of evaluation seed episodes.")
    parser.add_argument("--output-dir", type=str, default="outputs/eval")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained .pt checkpoint (required for RL models).")
    parser.add_argument("--resume", action="store_true", help="Skip seeds already recorded in per_seed CSV.")
    parser.add_argument("--seed-start", type=int, default=None, help="Starting evaluation seed (defaults to config evaluation.seed_start).")
    parser.add_argument("--num-seconds", type=int, default=3600)
    parser.add_argument("--config", type=str, default=str(REPO_ROOT / "configs/default.json"))
    parser.add_argument("--route-file", type=str, default=None, help="Optional custom SUMO .rou.xml override.")
    parser.add_argument("--save-telemetry", action="store_true", help="Export single-intersection telemetry for the first evaluated seed.")
    parser.add_argument("--gui", action="store_true")
    return parser.parse_args()


def _load_existing_rows(csv_path: Path) -> list[dict[str, Any]]:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return []
    with csv_path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    args = parse_args()
    cfg = load_default_config(args.config)
    model_cfg = MODEL_CONFIGS[args.model]
    is_rl = model_cfg.kind in ("vanilla_dqn", "d3qn")
    is_random_stress = args.demand == "random_stress"

    if model_cfg.requires_checkpoint and not args.checkpoint:
        raise ValueError(f"Model '{args.model}' requires --checkpoint <path/to/model.pt>.")

    seed_start = int(args.seed_start if args.seed_start is not None else cfg["evaluation"]["seed_start"])
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_stem = f"{args.model}_{args.network}_{args.demand}"
    per_seed_csv = out_dir / f"{run_stem}_per_seed.csv"
    summary_csv = out_dir / f"{run_stem}_summary.csv"

    existing_rows = _load_existing_rows(per_seed_csv) if args.resume else []
    completed_seeds = {int(row["seed"]) for row in existing_rows if "seed" in row}
    if not args.resume and per_seed_csv.exists():
        per_seed_csv.unlink()

    tmp_dir: Path | None = Path(tempfile.mkdtemp(prefix="sumo_eval_stress_")) if is_random_stress else None
    env = build_experiment_env(
        cfg=cfg,
        model_cfg=model_cfg,
        network=args.network,
        demand="balanced" if is_random_stress else args.demand,
        num_seconds=args.num_seconds,
        route_file_override=None if is_random_stress else args.route_file,
        use_gui=args.gui,
        validate_contracts=False,
    )

    try:
        num_actions = env.num_actions_map[env.ts_id]
        if is_random_stress:
            env.close()
            env = None

        controller = build_controller_or_agent(model_cfg, num_actions=num_actions, cfg=cfg)
        if is_rl and args.checkpoint:
            load_agent_checkpoint(args.checkpoint, controller, model_cfg)
            controller.online_net.eval()

        all_rows = list(existing_rows)
        for idx in range(args.episodes):
            seed = seed_start + idx
            if seed in completed_seeds:
                continue

            np.random.seed(seed)
            torch.manual_seed(seed)

            tmp_route_path: Path | None = None
            demand_label = args.demand
            if is_random_stress:
                assert tmp_dir is not None
                tmp_route_path = tmp_dir / f"stress_seed{seed}.rou.xml"
                gen_meta = generate_random_demand_route(
                    network=args.network,
                    output_path=tmp_route_path,
                    seed=seed,
                    num_seconds=args.num_seconds,
                )
                demand_label = f"random_stress:{gen_meta['archetype']}"
                if env is not None:
                    env.close()
                env = build_experiment_env(
                    cfg=cfg,
                    model_cfg=model_cfg,
                    network=args.network,
                    demand="balanced",
                    num_seconds=args.num_seconds,
                    route_file_override=str(tmp_route_path),
                    use_gui=args.gui,
                    validate_contracts=False,
                )

            assert env is not None
            t0 = time.time()
            result = run_episode(
                env=env,
                controller=controller,
                model_cfg=model_cfg,
                seed=seed,
                train=False,
                record_telemetry=bool(args.save_telemetry and idx == 0 and not env.is_multi_agent),
                episode_idx=idx + 1,
            )
            wall_s = time.time() - t0
            summary = result["metrics_summary"]

            if is_random_stress:
                env.close()
                env = None
                if tmp_route_path is not None:
                    tmp_route_path.unlink(missing_ok=True)

            row = build_eval_row(
                summary,
                model_name=model_cfg.display_name,
                network=args.network,
                demand_regime=demand_label,
                seed=seed,
            )
            append_csv_row(per_seed_csv, EPISODE_SUMMARY_COLUMNS, row)
            all_rows.append(row)
            write_summary_aggregates_csv(all_rows, summary_csv)

            recorder = result["recorder"]
            if recorder is not None:
                recorder.export(
                    output_dir=out_dir,
                    run_name=f"{run_stem}_seed{seed}",
                    model_name=model_cfg.display_name,
                    network=args.network,
                    demand_regime=demand_label,
                    seed=seed,
                )

            print(
                f"[{args.model} | {args.network}/{demand_label}] Seed {idx + 1:02d}/{args.episodes:02d} "
                f"(seed={seed}) | Delay={summary['avg_vehicle_delay_s_per_veh']:.2f}s | "
                f"MaxWait={summary['max_lane_wait_s']:.1f}s | "
                f"Thru={summary['throughput_veh_per_hr']:.1f}/hr | "
                f"CO2={summary['total_co2_g_per_km']:.1f}g/km ({wall_s:.1f}s)",
                flush=True,
            )

        print(f"\nSaved per-seed evaluation rows to: {per_seed_csv}", flush=True)
        print(f"Saved aggregate summary to: {summary_csv}", flush=True)
    finally:
        if env is not None:
            env.close()
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
