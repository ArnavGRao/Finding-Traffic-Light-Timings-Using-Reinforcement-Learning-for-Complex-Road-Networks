"""Shared CLI specifications, environment builders, checkpointing, and rollout runners."""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import sumo_rl
import torch

from src.agents import D3QNAgent
from src.baselines.fixed_time import FixedTimeController
from src.baselines.max_pressure import MaxPressureController
from src.baselines.vanilla_dqn import VanillaDQNAgent
from src.env.wrapper import SafetyShieldedTrafficEnv
from src.utils.contracts import EPISODE_SUMMARY_COLUMNS, OBS_DIM, load_default_config
from src.utils.metrics import METRIC_COLUMNS, TraCIMetricsLogger
from src.utils.telemetry import TelemetryRecorder

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ModelConfig:
    key: str
    display_name: str
    kind: str  # "fixed_time" | "max_pressure" | "vanilla_dqn" | "d3qn"
    use_shield: bool
    reward_mode: str
    dueling: bool = False
    double_dqn: bool = False
    requires_checkpoint: bool = False


MODEL_CONFIGS: dict[str, ModelConfig] = {
    "fixed_time": ModelConfig(
        key="fixed_time",
        display_name="Fixed-Time (30s)",
        kind="fixed_time",
        use_shield=True,
        reward_mode="starvation",
        requires_checkpoint=False,
    ),
    "max_pressure": ModelConfig(
        key="max_pressure",
        display_name="Max-Pressure",
        kind="max_pressure",
        use_shield=True,
        reward_mode="starvation",
        requires_checkpoint=False,
    ),
    "vanilla_dqn": ModelConfig(
        key="vanilla_dqn",
        display_name="Vanilla DQN",
        kind="vanilla_dqn",
        use_shield=False,
        reward_mode="queue_diff",
        requires_checkpoint=True,
    ),
    "d3qn_base": ModelConfig(
        key="d3qn_base",
        display_name="Model A (D3QN Base)",
        kind="d3qn",
        use_shield=False,
        reward_mode="delay_diff",
        dueling=True,
        double_dqn=True,
        requires_checkpoint=True,
    ),
    "d3qn_shield": ModelConfig(
        key="d3qn_shield",
        display_name="Model B (D3QN + Shield)",
        kind="d3qn",
        use_shield=True,
        reward_mode="delay_diff",
        dueling=True,
        double_dqn=True,
        requires_checkpoint=True,
    ),
    "d3qn_starvation": ModelConfig(
        key="d3qn_starvation",
        display_name="Model C (D3QN + Starvation)",
        kind="d3qn",
        use_shield=False,
        reward_mode="starvation",
        dueling=True,
        double_dqn=True,
        requires_checkpoint=True,
    ),
    "d3qn_proposed": ModelConfig(
        key="d3qn_proposed",
        display_name="Model D (D3QN Proposed)",
        kind="d3qn",
        use_shield=True,
        reward_mode="starvation",
        dueling=True,
        double_dqn=True,
        requires_checkpoint=True,
    ),
}

MODEL_CHOICES = tuple(MODEL_CONFIGS.keys())
NETWORK_CHOICES = ("single_intersection", "resco_4x4")
DEMAND_CHOICES = ("balanced", "asymmetric", "tidal")

NETWORK_SPECS: dict[str, dict[str, Any]] = {
    "single_intersection": {
        "net_file": "networks/single_intersection/single-intersection.net.xml",
        "single_agent": True,
        "routes": {
            "balanced": "networks/single_intersection/demand_balanced_symmetric.rou.xml",
            "asymmetric": "networks/single_intersection/demand_asymmetric_peak_arterial.rou.xml",
            "tidal": "networks/single_intersection/demand_stepwise_tidal.rou.xml",
        },
    },
    "resco_4x4": {
        "net_file": "networks/resco_4x4/grid4x4.net.xml",
        "single_agent": False,
        "routes": {
            "balanced": "networks/resco_4x4/grid4x4_1.rou.xml",
            "asymmetric": "networks/resco_4x4/grid4x4_1.rou.xml",
            "tidal": "networks/resco_4x4/grid4x4_1.rou.xml",
        },
    },
}


def resolve_network_paths(
    network: str,
    demand: str,
    route_file_override: str | None = None,
) -> tuple[Path, Path, bool]:
    """Resolve network XML, route XML, and single_agent flag."""
    if network not in NETWORK_SPECS:
        raise ValueError(f"Unsupported network '{network}'. Expected one of {NETWORK_CHOICES}.")
    if demand not in DEMAND_CHOICES:
        raise ValueError(f"Unsupported demand '{demand}'. Expected one of {DEMAND_CHOICES}.")

    spec = NETWORK_SPECS[network]
    net_path = REPO_ROOT / spec["net_file"]
    route_path = (
        Path(route_file_override)
        if route_file_override is not None
        else REPO_ROOT / spec["routes"][demand]
    )
    if not net_path.exists():
        raise FileNotFoundError(f"Network file not found: {net_path}")
    if not route_path.exists():
        raise FileNotFoundError(f"Route file not found: {route_path}")
    return net_path, route_path, bool(spec["single_agent"])


def build_experiment_env(
    cfg: dict[str, Any],
    model_cfg: ModelConfig,
    network: str,
    demand: str,
    num_seconds: int,
    route_file_override: str | None = None,
    use_gui: bool = False,
    validate_contracts: bool = False,
) -> SafetyShieldedTrafficEnv:
    """Create a SafetyShieldedTrafficEnv configured for the requested model, network, and demand."""
    net_path, route_path, single_agent = resolve_network_paths(
        network, demand, route_file_override=route_file_override
    )
    base_env = sumo_rl.SumoEnvironment(
        net_file=str(net_path),
        route_file=str(route_path),
        single_agent=single_agent,
        use_gui=use_gui,
        num_seconds=int(num_seconds),
        delta_time=int(cfg["shield"]["delta_time_s"]),
        yellow_time=0,
        min_green=0,
        sumo_warnings=False,
        additional_sumo_cmd="--no-step-log",
    )
    return SafetyShieldedTrafficEnv(
        base_env,
        cfg=cfg,
        use_shield=model_cfg.use_shield,
        reward_mode=model_cfg.reward_mode,
        validate_contracts=validate_contracts,
    )


def build_controller_or_agent(
    model_cfg: ModelConfig,
    num_actions: int,
    cfg: dict[str, Any],
    *,
    lr: float = 5e-4,
    gamma: float = 0.99,
    batch_size: int = 64,
    buffer_capacity: int = 100_000,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay_steps: int = 25_000,
    device: str | None = None,
) -> Any:
    """Instantiate a baseline controller or trainable RL agent."""
    if model_cfg.kind == "fixed_time":
        green_s = float(cfg["evaluation"]["fixed_time_green_s"])
        return FixedTimeController(num_actions=num_actions, green_duration_s=green_s)
    if model_cfg.kind == "max_pressure":
        return MaxPressureController(min_green_s=float(cfg["shield"]["min_green_s"]))
    if model_cfg.kind == "vanilla_dqn":
        return VanillaDQNAgent(
            state_dim=int(cfg["observation"]["state_dim"]),
            num_actions=num_actions,
            lr=lr,
            gamma=gamma,
            batch_size=batch_size,
            buffer_capacity=buffer_capacity,
            eps_start=eps_start,
            eps_end=eps_end,
            eps_decay_steps=eps_decay_steps,
            device=device,
        )
    if model_cfg.kind == "d3qn":
        return D3QNAgent(
            state_dim=int(cfg["observation"]["state_dim"]),
            num_actions=num_actions,
            lr=lr,
            gamma=gamma,
            batch_size=batch_size,
            buffer_capacity=buffer_capacity,
            dueling=model_cfg.dueling,
            double_dqn=model_cfg.double_dqn,
            device=device,
        )
    raise ValueError(f"Unsupported model kind: {model_cfg.kind}")


def save_agent_checkpoint(
    filepath: str | Path,
    agent: Any,
    model_cfg: ModelConfig,
    network: str,
    demand: str,
    episode: int,
    global_step: int,
    best_reward: float,
) -> None:
    """Save an RL checkpoint atomically so Colab disconnects do not corrupt .pt files."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    payload: dict[str, Any] = {
        "online_net": agent.online_net.state_dict(),
        "target_net": agent.target_net.state_dict(),
        "optimizer": agent.optimizer.state_dict(),
        "episode": int(episode),
        "global_step": int(global_step),
        "best_reward": float(best_reward),
        "model": model_cfg.key,
        "network": network,
        "demand": demand,
    }
    if model_cfg.kind == "d3qn":
        payload.update(
            {
                "train_steps": int(agent.train_steps),
                "dueling": bool(agent.dueling),
                "double_dqn": bool(agent.double_dqn),
            }
        )
    elif model_cfg.kind == "vanilla_dqn":
        payload["step_count"] = int(agent.step_count)

    torch.save(payload, str(tmp_path))
    os.replace(tmp_path, path)


def load_agent_checkpoint(
    filepath: str | Path,
    agent: Any,
    model_cfg: ModelConfig,
) -> dict[str, Any]:
    """Load an RL checkpoint into agent and return metadata (episode, global_step, best_reward)."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    device = getattr(agent, "device", "cpu")
    ckpt = torch.load(str(path), map_location=device, weights_only=True)
    agent.online_net.load_state_dict(ckpt["online_net"])
    if "target_net" in ckpt:
        agent.target_net.load_state_dict(ckpt["target_net"])
    else:
        agent.target_net.load_state_dict(ckpt["online_net"])
    if "optimizer" in ckpt and hasattr(agent, "optimizer"):
        agent.optimizer.load_state_dict(ckpt["optimizer"])

    if model_cfg.kind == "d3qn":
        agent.train_steps = int(ckpt.get("train_steps", 0))
    elif model_cfg.kind == "vanilla_dqn":
        agent.step_count = int(ckpt.get("step_count", ckpt.get("global_step", 0)))

    return {
        "episode": int(ckpt.get("episode", -1)),
        "global_step": int(ckpt.get("global_step", 0)),
        "best_reward": float(ckpt.get("best_reward", -float("inf"))),
    }


def append_csv_row(filepath: str | Path, fieldnames: list[str] | tuple[str, ...], row: dict[str, Any]) -> None:
    """Append a single CSV row (writing header if file is empty/missing) and flush to disk."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fieldnames))
        if write_header:
            writer.writeheader()
        writer.writerow({col: row[col] for col in fieldnames})
        stream.flush()
        os.fsync(stream.fileno())


def build_eval_row(
    metrics_summary: dict[str, Any],
    model_name: str,
    network: str,
    demand_regime: str,
    seed: int,
) -> dict[str, Any]:
    """Format a TraCIMetricsLogger summary into a row matching EPISODE_SUMMARY_COLUMNS."""
    row = {
        "model_name": model_name,
        "network": network,
        "demand_regime": demand_regime,
        "seed": int(seed),
    }
    for col in EPISODE_SUMMARY_COLUMNS:
        if col not in row:
            row[col] = float(metrics_summary[col])
    return row


def run_episode(
    env: SafetyShieldedTrafficEnv,
    controller: Any,
    model_cfg: ModelConfig,
    *,
    seed: int,
    train: bool = False,
    global_step: int = 0,
    warmup_steps: int = 500,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay_steps: int = 25_000,
    record_telemetry: bool = False,
    episode_idx: int = 1,
) -> dict[str, Any]:
    """Run one SUMO episode (single- or multi-agent) with TraCIMetricsLogger attached."""
    obs_out, info_out = env.reset(seed=seed)
    logger = TraCIMetricsLogger(env.unwrapped.sumo)
    env.set_substep_callback(logger.record_step)

    if env.is_multi_agent:
        obs_map = obs_out
        info_map = info_out
    else:
        obs_map = {env.ts_id: obs_out}
        info_map = {env.ts_id: info_out}

    recorder: TelemetryRecorder | None = None
    if record_telemetry and not env.is_multi_agent:
        recorder = TelemetryRecorder(episode=episode_idx, delta_time_s=env.delta_time_s)
        recorder.reset(episode=episode_idx, initial_info=info_map[env.ts_id])

    ep_reward = 0.0
    losses: list[float] = []
    blocked_switches = 0
    decision_steps = 0
    epsilon = 0.0
    done = False

    while not done:
        if train:
            decay_ratio = min(1.0, global_step / max(1, eps_decay_steps))
            epsilon = eps_start - decay_ratio * (eps_start - eps_end)

        action_map: dict[str, int] = {}
        for tid in env.ts_ids:
            obs = obs_map[tid]
            info = info_map[tid]
            if model_cfg.kind == "fixed_time":
                action = int(controller.select_action(info))
            elif model_cfg.kind == "max_pressure":
                action = int(controller.select_action(info, env))
            elif model_cfg.kind == "vanilla_dqn":
                action = int(controller.select_action(obs, deterministic=not train))
            elif model_cfg.kind == "d3qn":
                action = int(controller.act(obs, info["action_mask"], epsilon=epsilon if train else 0.0))
            else:
                raise ValueError(f"Unsupported model kind: {model_cfg.kind}")
            action_map[tid] = action

        if env.is_multi_agent:
            next_obs_map, reward_map, term_map, trunc_map, next_info_map = env.step(action_map)
            done = bool(term_map["__all__"] or trunc_map["__all__"])
        else:
            tid = env.ts_id
            n_obs, rew, term, trunc, n_info = env.step(action_map[tid])
            next_obs_map = {tid: n_obs}
            reward_map = {tid: rew}
            next_info_map = {tid: n_info}
            done = bool(term or trunc)

        step_reward_sum = float(sum(reward_map.values()))
        ep_reward += step_reward_sum / max(1, len(env.ts_ids))

        if recorder is not None:
            recorder.record_step(next_info_map[env.ts_id], reward_map[env.ts_id], episode=episode_idx)

        for tid in env.ts_ids:
            if not next_info_map[tid]["mask_approved"]:
                blocked_switches += 1

            if train:
                if model_cfg.kind == "vanilla_dqn":
                    controller.store_transition(
                        obs_map[tid],
                        action_map[tid],
                        reward_map[tid],
                        next_obs_map[tid],
                        float(done),
                    )
                elif model_cfg.kind == "d3qn":
                    controller.store_transition(
                        obs_map[tid],
                        action_map[tid],
                        reward_map[tid],
                        next_obs_map[tid],
                        done,
                        info_map[tid]["action_mask"],
                        next_info_map[tid]["action_mask"],
                    )

        if train:
            if model_cfg.kind == "vanilla_dqn":
                if len(controller.replay) >= max(controller.batch_size, warmup_steps):
                    loss = controller.train_step()
                    if loss is not None:
                        losses.append(float(loss))
            elif model_cfg.kind == "d3qn":
                if len(controller.replay_buffer) >= max(controller.batch_size, warmup_steps):
                    metrics = controller.train_step()
                    if metrics is not None:
                        losses.append(float(metrics["loss"]))

        obs_map = next_obs_map
        info_map = next_info_map
        decision_steps += 1
        global_step += 1

    env.set_substep_callback(None)
    summary = logger.summary()
    return {
        "total_reward": float(ep_reward),
        "mean_loss": float(np.mean(losses)) if losses else 0.0,
        "epsilon": float(controller.current_epsilon() if model_cfg.kind == "vanilla_dqn" and train else epsilon),
        "blocked_switches": int(blocked_switches),
        "decision_steps": int(decision_steps),
        "global_step": int(global_step),
        "metrics_summary": summary,
        "recorder": recorder,
    }


def write_summary_aggregates_csv(
    rows: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Compute and write mean/std aggregates across completed per-seed evaluation rows."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".csv.tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("metric", "mean", "std"))
        for col in METRIC_COLUMNS:
            vals = [float(r[col]) for r in rows if col in r]
            if vals:
                writer.writerow((col, float(np.mean(vals)), float(np.std(vals))))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp_path, path)
    return path
