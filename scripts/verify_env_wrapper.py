import json
import sumo_rl
from src.env.wrapper import SafetyShieldedTrafficEnv


def main():
    with open("configs/default.json") as f:
        cfg = json.load(f)

    ablation_configs = [
        ("Model A (Baseline D3QN)", False, False),
        ("Model B (D3QN + Shield)", True, False),
        ("Model C (D3QN + Starvation)", False, True),
        ("Model D (Complete Proposed)", True, True),
    ]

    for name, use_shield, use_starvation in ablation_configs:
        base_env = sumo_rl.SumoEnvironment(
            net_file="networks/single_intersection/single-intersection.net.xml",
            route_file="networks/single_intersection/single-intersection-vhvh.rou.xml",
            single_agent=True,
            use_gui=False,
            num_seconds=200,
            delta_time=5,
            yellow_time=0,
            min_green=0,
        )
        env = SafetyShieldedTrafficEnv(
            base_env,
            cfg=cfg,
            use_shield=use_shield,
            use_starvation_reward=use_starvation,
            validate_contracts=True,
        )
        try:
            obs, info = env.reset(seed=42)
            blocked_count = 0
            for _ in range(20):
                proposed = (info["active_phase"] + 1) % env.action_space.n
                obs, reward, terminated, truncated, info = env.step(proposed)
                if not info["mask_approved"]:
                    blocked_count += 1
            print(
                f"{name}: passed contract validation | blocked_switches={blocked_count} | last_reward={reward:.4f}"
            )
        finally:
            env.close()


if __name__ == "__main__":
    main()