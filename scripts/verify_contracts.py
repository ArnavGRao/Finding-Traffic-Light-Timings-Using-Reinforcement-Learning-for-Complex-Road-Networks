import numpy as np
from src.utils.contracts import (
    OBS_DIM,
    MASK_ALLOW,
    MASK_BLOCK,
    TELEMETRY_COLUMNS,
    EPISODE_SUMMARY_COLUMNS,
    load_default_config,
    validate_step_contract,
)


def main():
    cfg = load_default_config()
    assert cfg["observation"]["state_dim"] == OBS_DIM

    dummy_obs = np.zeros((OBS_DIM,), dtype=np.float32)
    dummy_mask = np.array([MASK_ALLOW, MASK_BLOCK, MASK_BLOCK, MASK_BLOCK], dtype=np.float32)
    dummy_info = {
        "sim_time": 5.0,
        "ts_id": "t",
        "active_phase": 0,
        "elapsed_green_s": 5.0,
        "proposed_action": 1,
        "executed_action": 0,
        "mask_approved": False,
        "phase_switched": False,
        "lane_ids": [f"lane_{i}" for i in range(8)],
        "lane_vehicle_counts": [0.0] * 8,
        "lane_queues": [0.0] * 8,
        "lane_delays": [0.0] * 8,
        "lane_max_waits": [0.0] * 8,
        "step_throughput": 0,
        "step_co2_mg": 0.0,
        "step_distance_m": 0.0,
    }

    validate_step_contract(dummy_obs, dummy_mask, dummy_info, num_green_phases=4)
    print(f"Contracts verified: obs={dummy_obs.shape}, mask={dummy_mask.shape}, telemetry_cols={len(TELEMETRY_COLUMNS)}, summary_cols={len(EPISODE_SUMMARY_COLUMNS)}")


if __name__ == "__main__":
    main()
