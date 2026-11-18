from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.contracts import load_default_config
from src.utils.controllers import (
    CONTROLLER_SPECS,
    ControllerAdapter,
    build_controller,
    infer_network_and_demand,
    make_traffic_env,
    resolve_controller_key,
)
from src.utils.telemetry import TelemetryRecorder


DEFAULT_NET_FILE = "networks/single_intersection/single-intersection.net.xml"
DEFAULT_ROUTE_FILE = "networks/single_intersection/single-intersection-vhvh.rou.xml"


@dataclass(frozen=True)
class RolloutResult:
    """Result bundle returned by programmatic or CLI rollout execution."""

    run_name: str
    summary: dict[str, Any]
    exported_files: dict[str, Path]
    num_steps: int


def run_rollout(
    controller: str | ControllerAdapter,
    config_path: str | Path = "configs/default.json",
    net_file: str | Path = DEFAULT_NET_FILE,
    route_file: str | Path = DEFAULT_ROUTE_FILE,
    checkpoint_path: str | Path | None = None,
    seed: int = 42,
    num_seconds: int = 3600,
    delta_time_s: int | None = None,
    green_duration_s: float | None = None,
    use_shield: bool | None = None,
    reward_mode: str | None = None,
    network_name: str | None = None,
    demand_regime: str | None = None,
    output_dir: str | Path = "outputs",
    run_name: str | None = None,
    file_format: str = "csv",
    episode: int = 1,
    allow_untrained: bool = False,
) -> RolloutResult:
    """Execute one headless SUMO episode with a pluggable controller and export telemetry."""
    cfg = load_default_config(str(config_path))
    inferred_net, inferred_demand = infer_network_and_demand(net_file, route_file)
    resolved_net = network_name or inferred_net
    resolved_demand = demand_regime or inferred_demand

    if isinstance(controller, str):
        ctrl_key = resolve_controller_key(controller)
        spec = CONTROLLER_SPECS[ctrl_key]
        resolved_shield = spec.default_use_shield if use_shield is None else bool(use_shield)
        resolved_reward = spec.default_reward_mode if reward_mode is None else str(reward_mode)
        model_label = spec.display_name
        resolved_run_name = (
            run_name
            or f"{ctrl_key}_{resolved_net}_{resolved_demand}_seed{int(seed)}"
        )
    else:
        ctrl_key = getattr(controller, "name", "custom_controller")
        resolved_shield = True if use_shield is None else bool(use_shield)
        resolved_reward = "starvation" if reward_mode is None else str(reward_mode)
        model_label = getattr(controller, "name", "CustomController")
        resolved_run_name = (
            run_name
            or f"{model_label.lower().replace(' ', '_')}_{resolved_net}_{resolved_demand}_seed{int(seed)}"
        )

    dt = int(delta_time_s if delta_time_s is not None else cfg["shield"]["delta_time_s"])
    env = make_traffic_env(
        cfg=cfg,
        net_file=net_file,
        route_file=route_file,
        use_shield=resolved_shield,
        reward_mode=resolved_reward,
        num_seconds=num_seconds,
        delta_time_s=dt,
        validate_contracts=True,
    )

    recorder = TelemetryRecorder(episode=episode, delta_time_s=float(dt))
    try:
        obs, info = env.reset(seed=int(seed))
        recorder.reset(episode=episode, initial_info=info)

        if isinstance(controller, str):
            adapter = build_controller(
                controller_key=ctrl_key,
                num_actions=env.action_space.n,
                cfg=cfg,
                checkpoint_path=checkpoint_path,
                green_duration_s=green_duration_s,
                allow_untrained=allow_untrained,
            )
        else:
            adapter = controller
            adapter.reset()

        done = False
        while not done:
            action = adapter.select_action(obs, info, env)
            obs, reward, terminated, truncated, info = env.step(action)
            recorder.record_step(info=info, reward=reward, episode=episode)
            done = bool(terminated or truncated)
    finally:
        env.close()

    metadata = {
        "controller_key": ctrl_key,
        "model_name": model_label,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "net_file": str(net_file),
        "route_file": str(route_file),
        "use_shield": resolved_shield,
        "reward_mode": resolved_reward,
        "seed": int(seed),
        "num_seconds": int(num_seconds),
        "min_green_s": float(cfg["shield"]["min_green_s"]),
        "yellow_s": float(cfg["shield"]["yellow_s"]),
        "all_red_s": float(cfg["shield"]["all_red_s"]),
        "w_thresh_s": float(cfg["reward"]["w_thresh_s"]),
    }

    exported = recorder.export(
        output_dir=output_dir,
        run_name=resolved_run_name,
        model_name=model_label,
        network=resolved_net,
        demand_regime=resolved_demand,
        seed=int(seed),
        file_format=file_format,
        metadata=metadata,
    )
    summary = recorder.compute_episode_summary(
        model_name=model_label,
        network=resolved_net,
        demand_regime=resolved_demand,
        seed=int(seed),
    )

    return RolloutResult(
        run_name=resolved_run_name,
        summary=summary,
        exported_files=exported,
        num_steps=len(recorder),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single SUMO rollout and export contract-compliant telemetry and summary logs."
    )
    parser.add_argument(
        "--controller",
        required=True,
        help=(
            "Controller key or alias: fixed_time, max_pressure, vanilla_dqn, "
            "d3qn_a (model_a), d3qn_b (model_b), d3qn_c (model_c), d3qn_d (model_d)."
        ),
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to PyTorch .pt checkpoint.")
    parser.add_argument(
        "--allow-untrained",
        action="store_true",
        help="Allow running vanilla_dqn without a checkpoint file for smoke testing.",
    )
    parser.add_argument("--config", type=str, default="configs/default.json", help="Path to default config JSON.")
    parser.add_argument("--network", type=str, default=DEFAULT_NET_FILE, help="Path to SUMO .net.xml file.")
    parser.add_argument("--route", type=str, default=DEFAULT_ROUTE_FILE, help="Path to SUMO .rou.xml file.")
    parser.add_argument("--network-name", type=str, default=None, help="Override network label in summary.")
    parser.add_argument("--demand-regime", type=str, default=None, help="Override demand regime label in summary.")
    parser.add_argument("--seed", type=int, default=42, help="SUMO simulation seed.")
    parser.add_argument("--num-seconds", type=int, default=3600, help="Episode duration in simulated seconds.")
    parser.add_argument("--delta-time", type=int, default=None, help="Decision interval in seconds (default 5).")
    parser.add_argument("--green-duration", type=float, default=None, help="Fixed-Time green phase duration (s).")
    shield_group = parser.add_mutually_exclusive_group()
    shield_group.add_argument("--use-shield", dest="use_shield", action="store_true", default=None)
    shield_group.add_argument("--no-shield", dest="use_shield", action="store_false")
    parser.add_argument(
        "--reward-mode",
        choices=["starvation", "delay_diff", "queue_diff"],
        default=None,
        help="Override reward mode (defaults to controller spec).",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "parquet", "both"],
        default="csv",
        help="Step telemetry export format.",
    )
    parser.add_argument("--output-dir", type=str, default="outputs", help="Destination directory for exports.")
    parser.add_argument("--run-name", type=str, default=None, help="Custom output filename prefix.")
    parser.add_argument("--episode", type=int, default=1, help="Episode index written into telemetry rows.")
    parser.add_argument("--quiet", action="store_true", help="Suppress stdout summary output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_rollout(
        controller=args.controller,
        config_path=args.config,
        net_file=args.network,
        route_file=args.route,
        checkpoint_path=args.checkpoint,
        seed=args.seed,
        num_seconds=args.num_seconds,
        delta_time_s=args.delta_time,
        green_duration_s=args.green_duration,
        use_shield=args.use_shield,
        reward_mode=args.reward_mode,
        network_name=args.network_name,
        demand_regime=args.demand_regime,
        output_dir=args.output_dir,
        run_name=args.run_name,
        file_format=args.format,
        episode=args.episode,
        allow_untrained=args.allow_untrained,
    )
    if not args.quiet:
        print(f"Completed rollout '{result.run_name}' ({result.num_steps} steps).")
        for key, path in result.exported_files.items():
            print(f"  - {key}: {path}")
        print("Episode summary:")
        for key, val in result.summary.items():
            print(f"  {key}: {val}")


if __name__ == "__main__":
    main()
