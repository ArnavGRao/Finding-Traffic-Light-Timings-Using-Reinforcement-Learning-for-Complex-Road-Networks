"""Integration smoke test across single-intersection and RESCO 4x4 networks.

Verifies:
1. Multi-agent SafetyShieldedTrafficEnv wrapping across all 16 RESCO 4x4 ts_ids
   with R^33 observations and action masks per agent.
2. All 4 ablation variants (Models A, B, C, D) on both networks for 500s episodes.
3. Action masks applied during both action selection and masked Double-DQN Y_t.
4. Prioritized Experience Replay updates executing cleanly on CUDA.
5. Checkpoint save/reload integrity.
6. All 7 evaluation metrics logged by src/utils/metrics.py (TraCIMetricsLogger).
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import sumo_rl
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agents import D3QNAgent
from src.env.wrapper import SafetyShieldedTrafficEnv
from src.utils.contracts import EPISODE_SUMMARY_COLUMNS, MASK_ALLOW, MASK_BLOCK, OBS_DIM, load_default_config
from src.utils.metrics import TraCIMetricsLogger

EVALUATION_METRICS = tuple(
    col for col in EPISODE_SUMMARY_COLUMNS
    if col not in ("model_name", "network", "demand_regime", "seed")
)
assert len(EVALUATION_METRICS) == 7

ABLATION_VARIANTS = {
    "Model A (Baseline D3QN)":      {"dueling": True, "double_dqn": True, "shield": False, "starvation": False},
    "Model B (D3QN + Shield)":      {"dueling": True, "double_dqn": True, "shield": True,  "starvation": False},
    "Model C (D3QN + Starvation)":  {"dueling": True, "double_dqn": True, "shield": False, "starvation": True},
    "Model D (Proposed Complete)":  {"dueling": True, "double_dqn": True, "shield": True,  "starvation": True},
}

NETWORK_CONFIGS = {
    "single_intersection": {
        "net_file": "networks/single_intersection/single-intersection.net.xml",
        "route_file": "networks/single_intersection/demand_balanced_symmetric.rou.xml",
        "single_agent": True,
        "expected_ts_count": 1,
        "expected_num_actions": 4,
    },
    "resco_4x4": {
        "net_file": "networks/resco_4x4/grid4x4.net.xml",
        "route_file": "networks/resco_4x4/grid4x4_1.rou.xml",
        "single_agent": False,
        "expected_ts_count": 16,
        "expected_num_actions": 8,
    },
}


def verify_masked_double_dqn_on_cuda(device: torch.device, num_actions: int) -> None:
    """Verify Double-DQN target Y_t ignores blocked (-1e9) actions on CUDA."""
    agent = D3QNAgent(
        state_dim=OBS_DIM,
        num_actions=num_actions,
        gamma=0.9,
        dueling=True,
        double_dqn=True,
        device=str(device),
    )
    batch_size = 8
    states = torch.randn(batch_size, OBS_DIM, device=device)
    actions = torch.zeros(batch_size, dtype=torch.int64, device=device)
    rewards = torch.full((batch_size,), 1.5, dtype=torch.float32, device=device)
    next_states = torch.randn(batch_size, OBS_DIM, device=device)
    dones = torch.zeros(batch_size, dtype=torch.float32, device=device)

    valid_action = num_actions - 1
    next_masks = torch.full((batch_size, num_actions), float(MASK_BLOCK), dtype=torch.float32, device=device)
    next_masks[:, valid_action] = float(MASK_ALLOW)

    with torch.no_grad():
        for p in agent.online_net.parameters():
            p.zero_()
        online_bias = torch.linspace(100.0, 10.0, num_actions, device=device)
        online_bias[valid_action] = -50.0  # lowest unmasked Q, but ONLY valid action
        agent.online_net.advantage_stream[-1].bias.copy_(online_bias)

        for p in agent.target_net.parameters():
            p.zero_()
        target_bias = torch.arange(1, num_actions + 1, dtype=torch.float32, device=device) * 10.0
        agent.target_net.advantage_stream[-1].bias.copy_(target_bias)
        expected_target_q = agent.target_net(next_states)[:, valid_action]

    is_weights = torch.ones(batch_size, dtype=torch.float32, device=device)
    loss, td_errors, y_t = agent.compute_target_and_loss(
        states, actions, rewards, next_states, dones, next_masks, is_weights
    )
    expected_y_t = rewards + 0.9 * expected_target_q
    assert y_t.device.type == device.type
    assert torch.allclose(y_t, expected_y_t, atol=1e-4), "Masked Double-DQN target Y_t mismatch"
    assert torch.isfinite(loss) and torch.all(torch.isfinite(td_errors))


def run_episode_smoke_test(
    network_name: str,
    net_cfg: dict,
    variant_name: str,
    variant: dict,
    cfg: dict,
    device: torch.device,
    num_seconds: int = 500,
) -> dict[str, float | int]:
    base_env = sumo_rl.SumoEnvironment(
        net_file=str(REPO_ROOT / net_cfg["net_file"]),
        route_file=str(REPO_ROOT / net_cfg["route_file"]),
        single_agent=net_cfg["single_agent"],
        use_gui=False,
        num_seconds=num_seconds,
        delta_time=int(cfg["shield"]["delta_time_s"]),
        yellow_time=0,
        min_green=0,
    )
    env = SafetyShieldedTrafficEnv(
        base_env,
        cfg=cfg,
        use_shield=variant["shield"],
        use_starvation_reward=variant["starvation"],
        validate_contracts=True,
    )

    try:
        assert len(env.ts_ids) == net_cfg["expected_ts_count"], (
            f"{network_name}: expected {net_cfg['expected_ts_count']} ts_ids, got {len(env.ts_ids)}"
        )
        assert env.is_multi_agent == (not net_cfg["single_agent"])
        num_actions = net_cfg["expected_num_actions"]
        for tid in env.ts_ids:
            assert env.num_actions_map[tid] == num_actions

        agent = D3QNAgent(
            state_dim=OBS_DIM,
            num_actions=num_actions,
            batch_size=32,
            buffer_capacity=4096,
            dueling=variant["dueling"],
            double_dqn=variant["double_dqn"],
            device=str(device),
        )
        assert agent.device.type == device.type

        obs_out, info_out = env.reset(seed=1000)
        logger = TraCIMetricsLogger(env.unwrapped.sumo)
        env.set_substep_callback(logger.record_step)

        if env.is_multi_agent:
            obs_map = obs_out
            info_map = info_out
        else:
            obs_map = {env.ts_id: obs_out}
            info_map = {env.ts_id: info_out}

        for tid in env.ts_ids:
            assert obs_map[tid].shape == (OBS_DIM,)
            assert info_map[tid]["action_mask"].shape == (num_actions,)

        mask_blocks_observed = 0
        train_metrics_list: list[dict[str, float]] = []
        step_count = 0
        done = False

        while not done:
            epsilon = max(0.05, 0.5 - 0.004 * step_count)
            action_map: dict[str, int] = {}
            for tid in env.ts_ids:
                mask = info_map[tid]["action_mask"]
                if np.any(np.isclose(mask, MASK_BLOCK)):
                    mask_blocks_observed += 1
                action = agent.act(obs_map[tid], mask, epsilon=epsilon)
                assert np.isclose(mask[action], MASK_ALLOW), (
                    f"{network_name}/{variant_name}/{tid}: selected blocked action {action} with mask {mask}"
                )
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

            for tid in env.ts_ids:
                assert next_obs_map[tid].shape == (OBS_DIM,)
                assert next_info_map[tid]["action_mask"].shape == (num_actions,)
                agent.store_transition(
                    state=obs_map[tid],
                    action=action_map[tid],
                    reward=reward_map[tid],
                    next_state=next_obs_map[tid],
                    done=done,
                    mask=info_map[tid]["action_mask"],
                    next_mask=next_info_map[tid]["action_mask"],
                )

            if len(agent.replay_buffer) >= agent.batch_size:
                train_metrics = agent.train_step()
                assert train_metrics is not None
                for key, val in train_metrics.items():
                    assert math.isfinite(val), f"Non-finite training metric {key}={val}"
                train_metrics_list.append(train_metrics)

            obs_map = next_obs_map
            info_map = next_info_map
            step_count += 1

        if variant["shield"]:
            assert mask_blocks_observed > 0, f"{variant_name} expected active shield blocks when elapsed_green < min_green"
        else:
            assert mask_blocks_observed == 0, f"{variant_name} should not block actions when shield is disabled"

        assert len(train_metrics_list) > 0, "Expected at least one PER + CUDA training update"
        assert agent.train_steps == len(train_metrics_list)
        assert agent.replay_buffer.tree.total() > 0.0

        summary = logger.summary()
        assert int(summary["steps"]) == num_seconds, f"Expected {num_seconds} TraCI substeps, got {summary['steps']}"
        for metric_name in EVALUATION_METRICS:
            assert metric_name in summary, f"Missing evaluation metric: {metric_name}"
            assert math.isfinite(float(summary[metric_name])), (
                f"Non-finite metric {metric_name}={summary[metric_name]}"
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "integration_ckpt.pt")
            agent.save(ckpt_path)
            reloaded = D3QNAgent(
                state_dim=OBS_DIM,
                num_actions=num_actions,
                dueling=variant["dueling"],
                double_dqn=variant["double_dqn"],
                device=str(device),
            )
            reloaded.load(ckpt_path)
            assert reloaded.train_steps == agent.train_steps
            for p_orig, p_load in zip(agent.online_net.parameters(), reloaded.online_net.parameters()):
                assert torch.equal(p_orig, p_load), "Checkpoint online_net mismatch"
            for p_orig, p_load in zip(agent.target_net.parameters(), reloaded.target_net.parameters()):
                assert torch.equal(p_orig, p_load), "Checkpoint target_net mismatch"
            sample_tid = env.ts_ids[0]
            reloaded_act = reloaded.act(obs_map[sample_tid], info_map[sample_tid]["action_mask"], epsilon=0.0)
            assert np.isclose(info_map[sample_tid]["action_mask"][reloaded_act], MASK_ALLOW)

        print(
            f"[PASS] {network_name:19s} | {variant_name:27s} | "
            f"ts={len(env.ts_ids):2d} | steps={step_count:3d} | "
            f"train_steps={agent.train_steps:3d} | last_loss={train_metrics_list[-1]['loss']:.4f} | "
            f"delay={summary['avg_vehicle_delay_s_per_veh']:.2f}s | "
            f"max_wait={summary['max_lane_wait_s']:.1f}s | "
            f"thru={summary['throughput_veh_per_hr']:.1f}/hr | "
            f"co2={summary['total_co2_g_per_km']:.1f}g/km"
        )
        return summary
    finally:
        env.close()


def main() -> None:
    np.random.seed(42)
    torch.manual_seed(42)

    assert torch.cuda.is_available(), "CUDA is required for integration verification on this host"
    device = torch.device("cuda")
    print(f"Running D3QN integration smoke test on device={device} ({torch.cuda.get_device_name(0)})")

    verify_masked_double_dqn_on_cuda(device, num_actions=4)
    verify_masked_double_dqn_on_cuda(device, num_actions=8)
    print("[PASS] Masked Double-DQN Y_t CUDA checks (4-action and 8-action).")

    cfg = load_default_config(str(REPO_ROOT / "configs/default.json"))
    for network_name, net_cfg in NETWORK_CONFIGS.items():
        for variant_name, variant in ABLATION_VARIANTS.items():
            run_episode_smoke_test(
                network_name=network_name,
                net_cfg=net_cfg,
                variant_name=variant_name,
                variant=variant,
                cfg=cfg,
                device=device,
                num_seconds=500,
            )

    print("\nAll 8 integration smoke tests (4 variants x 2 networks) passed cleanly.")


if __name__ == "__main__":
    main()
