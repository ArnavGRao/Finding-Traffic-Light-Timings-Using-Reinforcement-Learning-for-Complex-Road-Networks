from __future__ import annotations

import json
from pathlib import Path
from typing import List, TypedDict
import numpy as np

# Contract 1: Observation & Action-Mask Constants
OBS_DIM = 33
NUM_LANES = 8
MAX_PHASES = 8

SLICE_VEH_COUNTS = slice(0, 8)
SLICE_QUEUES = slice(8, 16)
SLICE_PHASE_ONEHOT = slice(16, 24)
SLICE_ELAPSED_GREEN = slice(24, 25)
SLICE_MAX_WAITS = slice(25, 33)

MASK_ALLOW = np.float32(0.0)
MASK_BLOCK = np.float32(-1e9)


def load_default_config(path: str = "configs/default.json") -> dict:
    with open(Path(path), "r", encoding="utf-8") as f:
        return json.load(f)


# Contract 2: Standardized Gymnasium Step `info` Dictionary
class StepInfo(TypedDict):
    sim_time: float
    ts_id: str
    active_phase: int
    elapsed_green_s: float
    proposed_action: int
    executed_action: int
    mask_approved: bool
    phase_switched: bool
    lane_ids: List[str]
    lane_vehicle_counts: List[float]   # length 8, raw vehicle counts
    lane_queues: List[float]           # length 8, halting vehicle counts
    lane_delays: List[float]           # length 8, average vehicle delay (s)
    lane_max_waits: List[float]        # length 8, longest-waiting vehicle (s)
    step_throughput: int               # vehicles cleared during decision interval
    step_co2_mg: float                 # total CO2 emitted during interval (mg)
    step_distance_m: float             # total vehicle distance traveled during interval (m)


REQUIRED_STEP_INFO_KEYS = set(StepInfo.__annotations__.keys())


# Contract 3A: Step-Level Telemetry Schema (consumed by dashboard/)
TELEMETRY_COLUMNS = [
    "episode",
    "sim_time",
    "ts_id",
    "active_phase",
    "elapsed_green_s",
    "proposed_action",
    "executed_action",
    "mask_approved",
    "phase_switched",
    "total_queue",
    "max_lane_wait_s",
    "lane_wait_variance_s2",
    "step_throughput",
    "step_co2_mg",
    "reward",
]


# Contract 3B: Episode-Level Evaluation Schema (7 metrics across 50 seeds)
EPISODE_SUMMARY_COLUMNS = [
    "model_name",
    "network",
    "demand_regime",
    "seed",
    "avg_vehicle_delay_s_per_veh",
    "max_lane_wait_s",
    "lane_wait_variance_s2",
    "max_queue_vehicles",
    "throughput_veh_per_hr",
    "total_co2_g_per_km",
    "oscillation_switches_per_hr",
]


def validate_step_contract(
    obs: np.ndarray,
    action_mask: np.ndarray,
    info: dict,
    num_green_phases: int,
) -> None:
    """Asserts that an environment step satisfies all three interface contracts."""
    if not isinstance(obs, np.ndarray) or obs.shape != (OBS_DIM,) or obs.dtype != np.float32:
        raise ValueError(f"obs must be float32 ndarray of shape ({OBS_DIM},), got {type(obs)} {getattr(obs, 'shape', None)} {getattr(obs, 'dtype', None)}")

    if not np.all(np.isfinite(obs)):
        raise ValueError("obs contains NaN or Inf values")

    if (
        not isinstance(action_mask, np.ndarray)
        or action_mask.shape != (num_green_phases,)
        or action_mask.dtype != np.float32
    ):
        raise ValueError(
            f"action_mask must be float32 ndarray of shape ({num_green_phases},), "
            f"got {getattr(action_mask, 'shape', None)} {getattr(action_mask, 'dtype', None)}"
        )

    valid_mask_vals = np.isclose(action_mask, MASK_ALLOW) | np.isclose(action_mask, MASK_BLOCK)
    if not np.all(valid_mask_vals):
        raise ValueError(f"action_mask must only contain {MASK_ALLOW} or {MASK_BLOCK}, got {action_mask}")

    missing_keys = REQUIRED_STEP_INFO_KEYS - set(info.keys())
    if missing_keys:
        raise KeyError(f"Step info missing required contract keys: {sorted(missing_keys)}")

    for list_key in ("lane_ids", "lane_vehicle_counts", "lane_queues", "lane_delays", "lane_max_waits"):
        if len(info[list_key]) != NUM_LANES:
            raise ValueError(f"info['{list_key}'] must have length {NUM_LANES}, got {len(info[list_key])}")
