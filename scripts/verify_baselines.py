import json
import sumo_rl
from src.baselines.fixed_time import FixedTimeController
from src.baselines.max_pressure import MaxPressureController
from src.baselines.vanilla_dqn import VanillaDQNAgent
from src.env.wrapper import SafetyShieldedTrafficEnv


def make_env(cfg: dict, use_shield: bool, reward_mode: str, num_seconds: int = 300):
    base_env = sumo_rl.SumoEnvironment(
        net_file="networks/single_intersection/single-intersection.net.xml",
        route_file="networks/single_intersection/single-intersection-vhvh.rou.xml",
        single_agent=True,
        use_gui=False,
        num_seconds=num_seconds,
        delta_time=5,
        yellow_time=0,
        min_green=0,
    )
    return SafetyShieldedTrafficEnv(
        base_env,
        cfg=cfg,
        use_shield=use_shield,
        reward_mode=reward_mode,
        validate_contracts=True,
    )


def main():
    with open("configs/default.json") as f:
        cfg = json.load(f)

    # 1. Fixed-Time 30s Baseline
    env = make_env(cfg, use_shield=True, reward_mode="starvation")
    try:
        ctrl = FixedTimeController(num_actions=env.action_space.n, green_duration_s=30.0)
        obs, info = env.reset(seed=42)
        switches = 0
        done = False
        while not done:
            action = ctrl.select_action(info)
            obs, reward, terminated, truncated, info = env.step(action)
            switches += int(info["phase_switched"])
            done = terminated or truncated
        print(f"FixedTime30s: sim_time={info['sim_time']}s | switches={switches}")
    finally:
        env.close()

    # 2. Max-Pressure Baseline
    env = make_env(cfg, use_shield=True, reward_mode="starvation")
    try:
        ctrl = MaxPressureController(min_green_s=cfg["shield"]["min_green_s"])
        obs, info = env.reset(seed=42)
        switches = 0
        done = False
        while not done:
            action = ctrl.select_action(info, env)
            obs, reward, terminated, truncated, info = env.step(action)
            switches += int(info["phase_switched"])
            done = terminated or truncated
        print(f"MaxPressure: sim_time={info['sim_time']}s | switches={switches}")
    finally:
        env.close()

    # 3. Vanilla DQN Baseline (unshielded, queue-diff reward)
    env = make_env(cfg, use_shield=False, reward_mode="queue_diff", num_seconds=500)
    try:
        agent = VanillaDQNAgent(
            state_dim=33,
            num_actions=env.action_space.n,
            batch_size=32,
            eps_decay_steps=100,
        )
        obs, info = env.reset(seed=42)
        losses = []
        done = False
        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.store_transition(obs, action, reward, next_obs, float(done))
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)
            obs = next_obs
        print(
            f"VanillaDQN: device={agent.device} | steps={agent.step_count} | "
            f"updates={len(losses)} | last_loss={losses[-1]:.4f}"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()