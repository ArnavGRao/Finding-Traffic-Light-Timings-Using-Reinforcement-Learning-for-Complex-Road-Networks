from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.utils.contracts import (
    EPISODE_SUMMARY_COLUMNS,
    NUM_LANES,
    TELEMETRY_COLUMNS,
)


@dataclass(frozen=True)
class StepRecord:
    """Single simulation decision-step record capturing contract and per-lane metrics."""

    episode: int
    sim_time: float
    ts_id: str
    active_phase: int
    elapsed_green_s: float
    proposed_action: int
    executed_action: int
    mask_approved: bool
    phase_switched: bool
    total_queue: float
    max_lane_wait_s: float
    lane_wait_variance_s2: float
    step_throughput: int
    step_co2_mg: float
    reward: float
    lane_ids: tuple[str, ...]
    lane_vehicle_counts: tuple[float, ...]
    lane_queues: tuple[float, ...]
    lane_delays: tuple[float, ...]
    lane_max_waits: tuple[float, ...]
    step_distance_m: float

    def to_telemetry_dict(self) -> dict[str, Any]:
        """Return a dict matching the exact schema defined in TELEMETRY_COLUMNS."""
        return {
            "episode": self.episode,
            "sim_time": self.sim_time,
            "ts_id": self.ts_id,
            "active_phase": self.active_phase,
            "elapsed_green_s": self.elapsed_green_s,
            "proposed_action": self.proposed_action,
            "executed_action": self.executed_action,
            "mask_approved": self.mask_approved,
            "phase_switched": self.phase_switched,
            "total_queue": self.total_queue,
            "max_lane_wait_s": self.max_lane_wait_s,
            "lane_wait_variance_s2": self.lane_wait_variance_s2,
            "step_throughput": self.step_throughput,
            "step_co2_mg": self.step_co2_mg,
            "reward": self.reward,
        }


@dataclass(frozen=True)
class PhaseAuditResult:
    """Reconstructed signal state timeline, green phase blocks, and shield audit metrics."""

    signal_intervals_df: pd.DataFrame
    green_blocks_df: pd.DataFrame
    action_audit_df: pd.DataFrame
    total_steps: int
    shield_interventions: int
    intervention_rate_pct: float
    total_switches: int
    min_green_violations: int


def validate_telemetry_columns(df: pd.DataFrame) -> None:
    """Validate that a DataFrame contains every column required by TELEMETRY_COLUMNS."""
    missing = [col for col in TELEMETRY_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Telemetry DataFrame missing required TELEMETRY_COLUMNS: {missing}"
        )


def validate_episode_summary_columns(summary: dict[str, Any] | pd.DataFrame) -> None:
    """Validate that an episode summary dict or DataFrame contains EPISODE_SUMMARY_COLUMNS."""
    cols = set(summary.columns) if isinstance(summary, pd.DataFrame) else set(summary.keys())
    missing = [col for col in EPISODE_SUMMARY_COLUMNS if col not in cols]
    if missing:
        raise ValueError(
            f"Episode summary missing required EPISODE_SUMMARY_COLUMNS: {missing}"
        )


def _infer_duration_s(sim_times: Sequence[float], default_dt_s: float = 5.0) -> float:
    """Infer covered simulation duration in seconds from monotonically increasing step timestamps."""
    if not sim_times:
        return 0.0
    times = np.asarray(sim_times, dtype=np.float64)
    if len(times) == 1:
        return float(times[0]) if 0.0 < times[0] <= default_dt_s * 2.0 else float(default_dt_s)
    diffs = np.diff(times)
    positive_diffs = diffs[diffs > 0]
    step_dt = float(np.median(positive_diffs)) if len(positive_diffs) > 0 else float(default_dt_s)
    return float((times[-1] - times[0]) + step_dt)


def compute_episode_kpis(
    telemetry_df: pd.DataFrame,
    lane_df: pd.DataFrame | None = None,
    step_distance_m: Sequence[float] | None = None,
    fallback_summary: dict[str, Any] | None = None,
    delta_time_s: float = 5.0,
) -> dict[str, float | None]:
    """Compute the 7 evaluation KPIs over a telemetry DataFrame (full episode or time slice)."""
    validate_telemetry_columns(telemetry_df)
    if telemetry_df.empty:
        return {
            "avg_vehicle_delay_s_per_veh": 0.0,
            "max_lane_wait_s": 0.0,
            "lane_wait_variance_s2": 0.0,
            "max_queue_vehicles": 0.0,
            "throughput_veh_per_hr": 0.0,
            "total_co2_g_per_km": 0.0,
            "oscillation_switches_per_hr": 0.0,
        }

    sim_times = telemetry_df["sim_time"].astype(float).tolist()
    duration_s = _infer_duration_s(sim_times, default_dt_s=delta_time_s)

    max_lane_wait_s = float(telemetry_df["max_lane_wait_s"].max())
    lane_wait_variance_s2 = float(telemetry_df["lane_wait_variance_s2"].mean())
    max_queue_vehicles = float(telemetry_df["total_queue"].max())

    total_throughput = float(telemetry_df["step_throughput"].sum())
    throughput_veh_per_hr = (total_throughput / duration_s) * 3600.0 if duration_s > 0 else 0.0

    total_switches = int(telemetry_df["phase_switched"].astype(bool).sum())
    oscillation_switches_per_hr = (total_switches / duration_s) * 3600.0 if duration_s > 0 else 0.0

    avg_vehicle_delay_s_per_veh: float | None = None
    if lane_df is not None and not lane_df.empty and {"sim_time", "vehicle_count", "delay_s"}.issubset(lane_df.columns):
        t_min, t_max = min(sim_times), max(sim_times)
        lane_slice = lane_df[(lane_df["sim_time"] >= t_min) & (lane_df["sim_time"] <= t_max)]
        if not lane_slice.empty:
            total_veh_obs = float(lane_slice["vehicle_count"].sum())
            if total_veh_obs > 0:
                weighted_delay = float((lane_slice["vehicle_count"] * lane_slice["delay_s"]).sum())
                avg_vehicle_delay_s_per_veh = weighted_delay / total_veh_obs
            else:
                avg_vehicle_delay_s_per_veh = 0.0
    if avg_vehicle_delay_s_per_veh is None and fallback_summary is not None:
        val = fallback_summary.get("avg_vehicle_delay_s_per_veh")
        if val is not None:
            avg_vehicle_delay_s_per_veh = float(val)

    total_co2_g_per_km: float | None = None
    if step_distance_m is not None and len(step_distance_m) > 0:
        dist_arr = np.asarray(step_distance_m, dtype=np.float64)
        total_dist_m = float(np.sum(dist_arr))
        total_co2_mg = float(telemetry_df["step_co2_mg"].sum())
        total_co2_g_per_km = (total_co2_mg / total_dist_m) if total_dist_m > 0 else 0.0
    elif fallback_summary is not None:
        val = fallback_summary.get("total_co2_g_per_km")
        if val is not None:
            total_co2_g_per_km = float(val)

    return {
        "avg_vehicle_delay_s_per_veh": avg_vehicle_delay_s_per_veh,
        "max_lane_wait_s": max_lane_wait_s,
        "lane_wait_variance_s2": lane_wait_variance_s2,
        "max_queue_vehicles": max_queue_vehicles,
        "throughput_veh_per_hr": throughput_veh_per_hr,
        "total_co2_g_per_km": total_co2_g_per_km,
        "oscillation_switches_per_hr": oscillation_switches_per_hr,
    }


def reconstruct_phase_audit(
    telemetry_df: pd.DataFrame,
    delta_time_s: float = 5.0,
    min_green_s: float = 10.0,
    yellow_s: float = 3.0,
    all_red_s: float = 2.0,
    phase_labels: dict[int, str] | None = None,
) -> PhaseAuditResult:
    """Reconstruct green, yellow, and all-red signal intervals and safety-shield audit records."""
    validate_telemetry_columns(telemetry_df)
    if telemetry_df.empty:
        empty_intervals = pd.DataFrame(
            columns=["start_s", "end_s", "duration_s", "signal_state", "active_phase", "phase_label"]
        )
        empty_blocks = pd.DataFrame(
            columns=[
                "block_id",
                "active_phase",
                "phase_label",
                "start_s",
                "end_s",
                "duration_s",
                "ended_by_switch",
                "meets_min_green",
                "violates_min_green",
            ]
        )
        empty_audit = pd.DataFrame(
            columns=[
                "sim_time",
                "active_phase",
                "elapsed_green_s",
                "proposed_action",
                "executed_action",
                "mask_approved",
                "phase_switched",
                "shield_intervened",
            ]
        )
        return PhaseAuditResult(
            signal_intervals_df=empty_intervals,
            green_blocks_df=empty_blocks,
            action_audit_df=empty_audit,
            total_steps=0,
            shield_interventions=0,
            intervention_rate_pct=0.0,
            total_switches=0,
            min_green_violations=0,
        )

    df = telemetry_df.sort_values("sim_time").reset_index(drop=True)
    labels = phase_labels or {}

    def _phase_name(phase_id: int) -> str:
        return labels.get(phase_id, f"Phase {phase_id}")

    first_t = float(df.loc[0, "sim_time"])
    prev_t = 0.0 if first_t <= delta_time_s * 1.5 else max(0.0, first_t - delta_time_s)

    intervals: list[dict[str, Any]] = []
    green_blocks: list[dict[str, Any]] = []

    first_switched = bool(df.loc[0, "phase_switched"])
    current_phase = int(df.loc[0, "active_phase"])
    block_phase = int(df.loc[0, "executed_action"]) if not first_switched else 0
    block_start = prev_t
    last_green_end = prev_t
    block_idx = 0

    for row in df.itertuples(index=False):
        t_end = float(row.sim_time)
        t_start = prev_t
        active_phase = int(row.active_phase)
        switched = bool(row.phase_switched)

        if switched:
            block_dur = max(0.0, last_green_end - block_start)
            green_blocks.append(
                {
                    "block_id": block_idx,
                    "active_phase": block_phase,
                    "phase_label": _phase_name(block_phase),
                    "start_s": block_start,
                    "end_s": last_green_end,
                    "duration_s": block_dur,
                    "ended_by_switch": True,
                    "meets_min_green": block_dur >= min_green_s,
                    "violates_min_green": block_dur < min_green_s,
                }
            )
            block_idx += 1

            yellow_end = min(t_end, t_start + yellow_s)
            if yellow_end > t_start:
                intervals.append(
                    {
                        "start_s": t_start,
                        "end_s": yellow_end,
                        "duration_s": yellow_end - t_start,
                        "signal_state": "Yellow (3s)",
                        "active_phase": active_phase,
                        "phase_label": f"Yellow ({_phase_name(block_phase)} -> {_phase_name(active_phase)})",
                    }
                )
            if t_end > yellow_end:
                intervals.append(
                    {
                        "start_s": yellow_end,
                        "end_s": t_end,
                        "duration_s": t_end - yellow_end,
                        "signal_state": "All-Red (2s)",
                        "active_phase": active_phase,
                        "phase_label": f"All-Red ({_phase_name(block_phase)} -> {_phase_name(active_phase)})",
                    }
                )

            block_phase = active_phase
            block_start = t_end
            last_green_end = t_end
        else:
            if t_end > t_start:
                intervals.append(
                    {
                        "start_s": t_start,
                        "end_s": t_end,
                        "duration_s": t_end - t_start,
                        "signal_state": "Green",
                        "active_phase": active_phase,
                        "phase_label": _phase_name(active_phase),
                    }
                )
            block_phase = active_phase
            last_green_end = t_end

        current_phase = active_phase
        prev_t = t_end

    if last_green_end > block_start:
        trailing_dur = last_green_end - block_start
        green_blocks.append(
            {
                "block_id": block_idx,
                "active_phase": current_phase,
                "phase_label": _phase_name(current_phase),
                "start_s": block_start,
                "end_s": last_green_end,
                "duration_s": trailing_dur,
                "ended_by_switch": False,
                "meets_min_green": trailing_dur >= min_green_s,
                "violates_min_green": False,
            }
        )

    audit_df = df[
        [
            "sim_time",
            "active_phase",
            "elapsed_green_s",
            "proposed_action",
            "executed_action",
            "mask_approved",
            "phase_switched",
        ]
    ].copy()
    audit_df["mask_approved"] = audit_df["mask_approved"].astype(bool)
    audit_df["phase_switched"] = audit_df["phase_switched"].astype(bool)
    audit_df["shield_intervened"] = ~audit_df["mask_approved"]

    total_steps = len(audit_df)
    shield_interventions = int(audit_df["shield_intervened"].sum())
    intervention_rate_pct = (100.0 * shield_interventions / total_steps) if total_steps > 0 else 0.0
    total_switches = int(audit_df["phase_switched"].sum())

    blocks_df = pd.DataFrame(green_blocks)
    min_green_violations = int(blocks_df["violates_min_green"].sum()) if not blocks_df.empty else 0

    return PhaseAuditResult(
        signal_intervals_df=pd.DataFrame(intervals),
        green_blocks_df=blocks_df,
        action_audit_df=audit_df,
        total_steps=total_steps,
        shield_interventions=shield_interventions,
        intervention_rate_pct=intervention_rate_pct,
        total_switches=total_switches,
        min_green_violations=min_green_violations,
    )


class TelemetryRecorder:
    """Reusable recorder collecting per-step contract telemetry and episode summaries."""

    def __init__(self, episode: int = 1, delta_time_s: float = 5.0):
        self.episode = int(episode)
        self.delta_time_s = float(delta_time_s)
        self.records: list[StepRecord] = []
        self.initial_sim_time: float = 0.0
        self.lane_ids: tuple[str, ...] = ()

    def reset(
        self,
        episode: int | None = None,
        initial_info: dict[str, Any] | None = None,
    ) -> None:
        """Reset recorded buffers for a new episode."""
        if episode is not None:
            self.episode = int(episode)
        self.records.clear()
        self.initial_sim_time = (
            float(initial_info.get("sim_time", 0.0)) if initial_info else 0.0
        )
        self.lane_ids = (
            tuple(str(l) for l in initial_info.get("lane_ids", ()))
            if initial_info
            else ()
        )

    def record_step(
        self,
        info: dict[str, Any],
        reward: float,
        episode: int | None = None,
    ) -> StepRecord:
        """Collect one step record from SafetyShieldedTrafficEnv info and reward."""
        ep = self.episode if episode is None else int(episode)
        lane_ids = tuple(str(x) for x in info["lane_ids"])
        if len(lane_ids) != NUM_LANES:
            raise ValueError(f"Expected {NUM_LANES} lanes in info['lane_ids'], got {len(lane_ids)}")
        if not self.lane_ids:
            self.lane_ids = lane_ids

        counts = tuple(float(x) for x in info["lane_vehicle_counts"])
        queues = tuple(float(x) for x in info["lane_queues"])
        delays = tuple(float(x) for x in info["lane_delays"])
        max_waits = tuple(float(x) for x in info["lane_max_waits"])

        total_queue = float(np.sum(queues))
        max_lane_wait = float(np.max(max_waits)) if max_waits else 0.0
        lane_wait_var = float(np.var(max_waits)) if max_waits else 0.0

        rec = StepRecord(
            episode=ep,
            sim_time=float(info["sim_time"]),
            ts_id=str(info["ts_id"]),
            active_phase=int(info["active_phase"]),
            elapsed_green_s=float(info["elapsed_green_s"]),
            proposed_action=int(info["proposed_action"]),
            executed_action=int(info["executed_action"]),
            mask_approved=bool(info["mask_approved"]),
            phase_switched=bool(info["phase_switched"]),
            total_queue=total_queue,
            max_lane_wait_s=max_lane_wait,
            lane_wait_variance_s2=lane_wait_var,
            step_throughput=int(info["step_throughput"]),
            step_co2_mg=float(info["step_co2_mg"]),
            reward=float(reward),
            lane_ids=lane_ids,
            lane_vehicle_counts=counts,
            lane_queues=queues,
            lane_delays=delays,
            lane_max_waits=max_waits,
            step_distance_m=float(info["step_distance_m"]),
        )
        self.records.append(rec)
        return rec

    def __len__(self) -> int:
        return len(self.records)

    def telemetry_dataframe(self) -> pd.DataFrame:
        """Build a DataFrame with the exact columns and order of TELEMETRY_COLUMNS."""
        if not self.records:
            return pd.DataFrame(columns=list(TELEMETRY_COLUMNS))
        df = pd.DataFrame([r.to_telemetry_dict() for r in self.records])
        return df[list(TELEMETRY_COLUMNS)]

    def lane_dataframe(self) -> pd.DataFrame:
        """Build a long-form DataFrame of per-lane vehicle counts, queues, delays, and max waits."""
        cols = [
            "episode",
            "sim_time",
            "ts_id",
            "lane_idx",
            "lane_id",
            "vehicle_count",
            "queue",
            "delay_s",
            "max_wait_s",
        ]
        if not self.records:
            return pd.DataFrame(columns=cols)
        rows: list[dict[str, Any]] = []
        for rec in self.records:
            for idx in range(len(rec.lane_ids)):
                rows.append(
                    {
                        "episode": rec.episode,
                        "sim_time": rec.sim_time,
                        "ts_id": rec.ts_id,
                        "lane_idx": idx,
                        "lane_id": rec.lane_ids[idx],
                        "vehicle_count": rec.lane_vehicle_counts[idx],
                        "queue": rec.lane_queues[idx],
                        "delay_s": rec.lane_delays[idx],
                        "max_wait_s": rec.lane_max_waits[idx],
                    }
                )
        return pd.DataFrame(rows, columns=cols)

    def step_distances_m(self) -> list[float]:
        """Return step-aligned vehicle travel distances (m) for CO2 g/km slice computations."""
        return [rec.step_distance_m for rec in self.records]

    def compute_episode_summary(
        self,
        model_name: str,
        network: str,
        demand_regime: str,
        seed: int,
    ) -> dict[str, Any]:
        """Compute episode summary dictionary adhering strictly to EPISODE_SUMMARY_COLUMNS."""
        t_df = self.telemetry_dataframe()
        l_df = self.lane_dataframe()
        kpis = compute_episode_kpis(
            t_df,
            lane_df=l_df,
            step_distance_m=self.step_distances_m(),
            delta_time_s=self.delta_time_s,
        )
        summary = {
            "model_name": str(model_name),
            "network": str(network),
            "demand_regime": str(demand_regime),
            "seed": int(seed),
            "avg_vehicle_delay_s_per_veh": float(kpis["avg_vehicle_delay_s_per_veh"] or 0.0),
            "max_lane_wait_s": float(kpis["max_lane_wait_s"] or 0.0),
            "lane_wait_variance_s2": float(kpis["lane_wait_variance_s2"] or 0.0),
            "max_queue_vehicles": float(kpis["max_queue_vehicles"] or 0.0),
            "throughput_veh_per_hr": float(kpis["throughput_veh_per_hr"] or 0.0),
            "total_co2_g_per_km": float(kpis["total_co2_g_per_km"] or 0.0),
            "oscillation_switches_per_hr": float(kpis["oscillation_switches_per_hr"] or 0.0),
        }
        validate_episode_summary_columns(summary)
        return summary

    def summary_dataframe(
        self,
        model_name: str,
        network: str,
        demand_regime: str,
        seed: int,
    ) -> pd.DataFrame:
        """Return single-row DataFrame matching EPISODE_SUMMARY_COLUMNS."""
        summary = self.compute_episode_summary(model_name, network, demand_regime, seed)
        return pd.DataFrame([summary], columns=list(EPISODE_SUMMARY_COLUMNS))

    def export(
        self,
        output_dir: str | Path,
        run_name: str,
        model_name: str,
        network: str,
        demand_regime: str,
        seed: int,
        file_format: str = "csv",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Path]:
        """Export step telemetry (CSV/Parquet), per-lane metrics, and summary CSV/JSON."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        fmt = file_format.lower()
        if fmt not in {"csv", "parquet", "both"}:
            raise ValueError(f"Unsupported export format '{file_format}'. Use 'csv', 'parquet', or 'both'.")

        telemetry_df = self.telemetry_dataframe()
        validate_telemetry_columns(telemetry_df)
        lane_df = self.lane_dataframe()
        summary = self.compute_episode_summary(model_name, network, demand_regime, seed)
        summary_df = pd.DataFrame([summary], columns=list(EPISODE_SUMMARY_COLUMNS))

        exported: dict[str, Path] = {}

        if fmt in {"csv", "both"}:
            t_csv = out_dir / f"{run_name}_telemetry.csv"
            telemetry_df.to_csv(t_csv, index=False)
            exported["telemetry_csv"] = t_csv

            l_csv = out_dir / f"{run_name}_lanes.csv"
            lane_df.to_csv(l_csv, index=False)
            exported["lanes_csv"] = l_csv

        if fmt in {"parquet", "both"}:
            t_pq = out_dir / f"{run_name}_telemetry.parquet"
            telemetry_df.to_parquet(t_pq, index=False)
            exported["telemetry_parquet"] = t_pq

            l_pq = out_dir / f"{run_name}_lanes.parquet"
            lane_df.to_parquet(l_pq, index=False)
            exported["lanes_parquet"] = l_pq

        s_csv = out_dir / f"{run_name}_summary.csv"
        summary_df.to_csv(s_csv, index=False)
        exported["summary_csv"] = s_csv

        payload = {
            "summary": summary,
            "metadata": {
                **(metadata or {}),
                "run_name": run_name,
                "num_steps": len(self.records),
                "initial_sim_time": self.initial_sim_time,
                "delta_time_s": self.delta_time_s,
                "lane_ids": list(self.lane_ids),
                "telemetry_files": {
                    k: str(v.name) for k, v in exported.items() if k.startswith("telemetry_")
                },
                "lane_files": {
                    k: str(v.name) for k, v in exported.items() if k.startswith("lanes_")
                },
            },
            "step_distance_m": self.step_distances_m(),
        }
        s_json = out_dir / f"{run_name}_summary.json"
        with open(s_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        exported["summary_json"] = s_json

        return exported
