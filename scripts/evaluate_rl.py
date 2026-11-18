import argparse
import csv
import json
import os
import numpy as np
import sumo_rl
import torch

from src.agents import D3QNAgent
from src.env.wrapper import SafetyShieldedTrafficEnv
from src.utils.contracts import EPISODE_SUMMARY_COLUMNS, TELEMETRY_COLUMNS

MODEL_VARIANTS = {
    "vanilla_dqn": {"dueling": False, "double_dqn": False, "shield": False, "starvation": False},
    "model_a":     {"dueling": True,  "double_dqn": True,  "shield": False, "starvation": False},
    "model_b":     {"dueling": True,  "double_dqn": True,  "shield": True,  "starvation": False},
    "model_c":     {"dueling": True,  "double_dqn": True,  "shield": False, "starvation": True},
    "model_d":     {"dueling": True,  "double_dqn": True,  "shield": True,  "starvation": True},
}


def align_row_to_contract(metrics: dict, contract_columns: list[str]) -> dict:
    aliases = {
        "controller": metrics["model"],
        "model": metrics["model"],
        "model_name": metrics["model"],
        "regime": metrics["regime"],
        "demand_regime": metrics["regime"],
        "seed": metrics["seed"],
        "avg_delay_s": metrics["avg_delay_s"],
        "mean_delay_s": metrics["avg_delay_s"],
        "max_wait_s": metrics["max_wait_s"],
        "peak_wait_s": metrics["max_wait_s"],
        "avg_queue_length": metrics["avg_queue_length"],
        "mean_queue_length": metrics["avg_queue_length"],
        "total_throughput": metrics["total_throughput"],
        "throughput": metrics["total_throughput"],
        "total_co2_kg": metrics["total_co2_kg"],
        "co2_kg": metrics["total_co2_kg"],
        "total_co2_mg": metrics["total_co2_mg"],
        "shield_interventions": metrics["shield_interventions"],
        "blocked_switches": metrics["shield_interventions"],
        "total_reward": metrics["total_reward"],
        "episode_reward": metrics["total_reward"],
    }
    return {col: aliases.get(col, metrics.get(col, 0.0)) for col in contract_columns}


def main():
    parser = argparse.ArgumentParser(description="50-Seed Deterministic Evaluation for RL Traffic Agents")
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_VARIANTS.keys()))
    parser.add_argument("--regime", type=str, default="balanced")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained .pt checkpoint")
    parser.add_argument("--config", type=str, default="configs/default.json")
    parser.add_argument("--net-file", type=str, default="networks/single_intersection/single-intersection.net.xml")
    parser.add_argument("--route-file", type=str, default="networks/single_intersection/single-intersection-vhvh.rou.xml")
    parser.add_argument("--start-seed", type=int, default=1000)
    parser.add_argument("--num-seeds", type=int, default=50)
    parser.add_argument("--num-seconds", type=int, default=3600)
    parser.add_argument("--output-dir", type=str, default="outputs/eval")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    variant = MODEL_VARIANTS[args.model]
    os.makedirs(args.output_dir, exist_ok=True)

    base_env = sumo_rl.SumoEnvironment(
        net_file=args.net_file,
        route_file=args.route_file,
        single_agent=True,
        use_gui=False,
        num_seconds=args.num_seconds,
        delta_time=int(cfg["shield"]["delta_time_s"]),
        yellow_time=0,
        min_green=0,
    )
    env = SafetyShieldedTrafficEnv(
        base_env,
        cfg=cfg,
        use_shield=variant["shield"],
        use_starvation_reward=variant["starvation"],
        validate_contracts=False,
    )

    agent = D3QNAgent(
        state_dim=33,
        num_actions=env.action_space.n,
        dueling=variant["dueling"],
        double_dqn=variant["double_dqn"],
    )
    ckpt = torch.load(args.checkpoint, map_location=agent.device, weights_only=True)
    agent.online_net.load_state_dict(ckpt["online_net"])
    agent.online_net.eval()

    summary_csv = os.path.join(args.output_dir, f"eval_{args.model}_{args.regime}.csv")
    telemetry_csv = os.path.join(args.output_dir, f"telemetry_{args.model}_{args.regime}_seed{args.start_seed}.csv")

    summary_rows = []
    telemetry_rows = []

    for i in range(args.num_seeds):
        seed = args.start_seed + i
        obs, info = env.reset(seed=seed)
        mask = info["action_mask"]

        ep_reward = 0.0
        delays = []
        queues = []
        max_wait_s = 0.0
        total_throughput = 0
        total_co2_mg = 0.0
        shield_interventions = 0

        done = False
        while not done:
            action = agent.act(obs, mask, epsilon=0.0)
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            done = bool(terminated or truncated)

            ep_reward += reward
            step_delay = float(np.mean(next_info["lane_delays"]))
            step_queue = float(np.mean(next_info["lane_queues"]))
            step_max_wait = float(np.max(next_info["lane_max_waits"]))

            delays.append(step_delay)
            queues.append(step_queue)
            max_wait_s = max(max_wait_s, step_max_wait)
            total_throughput += int(next_info["step_throughput"])
            total_co2_mg += float(next_info["step_co2_mg"])
            if not next_info["mask_approved"]:
                shield_interventions += 1

            if i == 0:
                t_row = {col: next_info.get(col, 0) for col in TELEMETRY_COLUMNS}
                t_row.update(
                    {
                        "model": args.model,
                        "regime": args.regime,
                        "seed": seed,
                        "sim_time": next_info["sim_time"],
                        "active_phase": next_info["active_phase"],
                        "elapsed_green_s": next_info["elapsed_green_s"],
                        "proposed_action": next_info["proposed_action"],
                        "executed_action": next_info["executed_action"],
                        "mask_approved": int(next_info["mask_approved"]),
                        "avg_delay_s": round(step_delay, 3),
                        "avg_queue": round(step_queue, 3),
                        "max_wait_s": round(step_max_wait, 3),
                        "step_throughput": next_info["step_throughput"],
                        "step_co2_mg": round(next_info["step_co2_mg"], 2),
                        "reward": round(reward, 4),
                    }
                )
                telemetry_rows.append({col: t_row[col] for col in TELEMETRY_COLUMNS})

            obs = next_obs
            mask = next_info["action_mask"]

        raw_metrics = {
            "model": args.model,
            "regime": args.regime,
            "seed": seed,
            "avg_delay_s": round(float(np.mean(delays)), 4),
            "max_wait_s": round(max_wait_s, 4),
            "avg_queue_length": round(float(np.mean(queues)), 4),
            "total_throughput": int(total_throughput),
            "total_co2_mg": round(total_co2_mg, 2),
            "total_co2_kg": round(total_co2_mg / 1_000_000.0, 4),
            "shield_interventions": int(shield_interventions),
            "total_reward": round(ep_reward, 4),
        }
        summary_rows.append(align_row_to_contract(raw_metrics, list(EPISODE_SUMMARY_COLUMNS)))

        if (i + 1) % 10 == 0 or (i + 1) == args.num_seeds:
            print(
                f"[{args.model} | {args.regime}] Seed {i+1:02d}/{args.num_seeds:02d} -> "
                f"Delay={raw_metrics['avg_delay_s']:.2f}s | MaxWait={raw_metrics['max_wait_s']:.1f}s | "
                f"Thru={raw_metrics['total_throughput']}"
            )

    env.close()

    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(EPISODE_SUMMARY_COLUMNS))
        writer.writeheader()
        writer.writerows(summary_rows)

    if telemetry_rows:
        with open(telemetry_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(TELEMETRY_COLUMNS))
            writer.writeheader()
            writer.writerows(telemetry_rows)

    print(f"\nSaved 50-seed summary to: {summary_csv}")
    print(f"Saved sample telemetry to: {telemetry_csv}")


if __name__ == "__main__":
    main()
