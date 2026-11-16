import argparse
import sumo_rl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true", help="Launch sumo-gui for visual check")
    args = parser.parse_args()

    step_scale = 10 if args.gui else 1

    env = sumo_rl.SumoEnvironment(
        net_file="networks/single_intersection/single-intersection.net.xml",
        route_file="networks/single_intersection/single-intersection-vhvh.rou.xml",
        single_agent=True,
        use_gui=args.gui,
        num_seconds=200,
        delta_time=5 * step_scale,
        yellow_time=3 * step_scale,
        min_green=10 * step_scale,
        additional_sumo_cmd="--step-length 0.1 --delay 150" if args.gui else "",
    )

    obs, info = env.reset(seed=42)
    ts_id = env.ts_ids[0]
    ts = env.traffic_signals[ts_id]

    print(f"Traffic Signal ID: {ts_id}")
    print(f"Incoming Lanes ({len(ts.lanes)}): {ts.lanes}")
    print(f"Green Phases ({ts.num_green_phases}): {ts.green_phases}")
    print(f"All Phases ({len(ts.all_phases)}): {ts.all_phases}")
    print(f"Default Observation Shape: {obs.shape}")

    for step in range(40):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    if args.gui:
        input("Press Enter in terminal to close sumo-gui...")

    env.close()
    print("Environment step verification passed.")


if __name__ == "__main__":
    main()