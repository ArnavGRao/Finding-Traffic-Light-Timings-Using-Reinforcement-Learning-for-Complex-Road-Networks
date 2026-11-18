"""TraCI episode metrics and reproducible seed-sweep aggregation.

Call ``record_step`` once after every ``simulationStep``. The logger observes
all non-internal incoming lanes at every traffic light in the connected SUMO
simulation, so it works for both a single junction and a multi-signal network.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Callable, Mapping, Sequence

from src.utils.contracts import EPISODE_SUMMARY_COLUMNS


METRIC_COLUMNS = (
    "avg_vehicle_delay_s_per_veh",
    "avg_travel_time_s_per_veh",
    "max_lane_wait_s",
    "lane_wait_variance_s2",
    "max_queue_vehicles",
    "throughput_veh_per_hr",
    "total_co2_g",
    "total_co2_g_per_km",
    "oscillation_switches_per_hr",
)


class TraCIMetricsLogger:
    """Accumulate proposal-aligned metrics from an active TraCI connection."""

    def __init__(
        self,
        traci: Any,
        *,
        delta_time_s: float | None = None,
        traffic_light_ids: Sequence[str] | None = None,
    ) -> None:
        self.traci = traci
        self.delta_time_s = float(delta_time_s or traci.simulation.getDeltaT())
        self.traffic_light_ids = list(
            traffic_light_ids if traffic_light_ids is not None
            else traci.trafficlight.getIDList()
        )
        self.incoming_lanes: dict[str, list[str]] = {}
        for tls_id in self.traffic_light_ids:
            lanes = traci.trafficlight.getControlledLanes(tls_id)
            self.incoming_lanes[tls_id] = sorted(
                {lane for lane in lanes if lane and not lane.startswith(":")}
            )

        self.duration_s = 0.0
        self._start_time_s = float(traci.simulation.getTime())
        self.steps = 0
        self.throughput = 0
        self.total_co2_mg = 0.0
        self.total_distance_m = 0.0
        self.max_queue = 0
        self.max_lane_wait_s = 0.0
        self._lane_mean_wait_samples: list[float] = []
        self._last_time_loss: dict[str, float] = {}
        self._departure_time: dict[str, float] = {}
        self._arrived_delay_s = 0.0
        self._arrived_travel_time_s = 0.0
        self._last_green_state: dict[str, str] = {}
        self.switches = 0

    def record_step(self) -> None:
        """Record measurements for the latest completed SUMO step."""
        traci = self.traci
        dt = self.delta_time_s
        self.steps += 1
        self.duration_s += dt
        simulation_time = float(traci.simulation.getTime())

        for vehicle_id in traci.vehicle.getIDList():
            try:
                self._departure_time.setdefault(vehicle_id, float(traci.vehicle.getDeparture(vehicle_id)))
                self._last_time_loss[vehicle_id] = float(
                    traci.vehicle.getTimeLoss(vehicle_id)
                )
                self.total_co2_mg += float(
                    traci.vehicle.getCO2Emission(vehicle_id)
                ) * dt
                self.total_distance_m += float(
                    traci.vehicle.getSpeed(vehicle_id)
                ) * dt
            except Exception:
                # A vehicle can disappear between TraCI list and getter calls.
                continue

        arrived = list(traci.simulation.getArrivedIDList())
        self.throughput += len(arrived)
        for vehicle_id in arrived:
            self._arrived_delay_s += self._last_time_loss.pop(vehicle_id, 0.0)
            departure_time = self._departure_time.pop(vehicle_id, None)
            if departure_time is not None:
                self._arrived_travel_time_s += max(0.0, simulation_time - departure_time)

        step_lane_means: list[float] = []
        for lanes in self.incoming_lanes.values():
            intersection_queue = 0
            for lane_id in lanes:
                vehicle_ids = traci.lane.getLastStepVehicleIDs(lane_id)
                intersection_queue += int(traci.lane.getLastStepHaltingNumber(lane_id))
                waits = []
                for vehicle_id in vehicle_ids:
                    try:
                        waits.append(float(traci.vehicle.getWaitingTime(vehicle_id)))
                    except Exception:
                        continue
                if waits:
                    self.max_lane_wait_s = max(self.max_lane_wait_s, max(waits))
                    step_lane_means.append(fmean(waits))
                else:
                    step_lane_means.append(0.0)
            self.max_queue = max(self.max_queue, intersection_queue)
        if step_lane_means:
            self._lane_mean_wait_samples.extend(step_lane_means)

        for tls_id in self.traffic_light_ids:
            try:
                state = str(traci.trafficlight.getRedYellowGreenState(tls_id))
            except Exception:
                continue
            # Count only changes between active green configurations; yellow
            # and all-red clearance states are not separate policy switches.
            if not any(char in "Gg" for char in state) or any(char in "yY" for char in state):
                continue
            previous = self._last_green_state.get(tls_id)
            if previous is not None and previous != state:
                self.switches += 1
            self._last_green_state[tls_id] = state

    def summary(self) -> dict[str, float | int]:
        """Return episode metrics with explicit units and safe empty-run values."""
        total_delay = self._arrived_delay_s + sum(self._last_time_loss.values())
        observed_vehicles = self.throughput + len(self._last_time_loss)
        unfinished_travel_time = sum(
            max(0.0, self._start_time_s + self.duration_s - departure_time)
            for departure_time in self._departure_time.values()
        )
        total_travel_time = self._arrived_travel_time_s + unfinished_travel_time
        variance = (
            _population_variance(self._lane_mean_wait_samples)
            if self._lane_mean_wait_samples else 0.0
        )
        co2_g = self.total_co2_mg / 1_000_000.0
        return {
            "avg_vehicle_delay_s_per_veh": (
                total_delay / observed_vehicles if observed_vehicles else 0.0
            ),
            "avg_travel_time_s_per_veh": (
                total_travel_time / observed_vehicles if observed_vehicles else 0.0
            ),
            "max_lane_wait_s": self.max_lane_wait_s,
            "lane_wait_variance_s2": variance,
            "max_queue_vehicles": self.max_queue,
            "throughput_veh_per_hr": (
                self.throughput * 3600.0 / self.duration_s if self.duration_s else 0.0
            ),
            "total_co2_g": co2_g,
            # mg / m has the same numeric value as g / km.
            "total_co2_g_per_km": (
                self.total_co2_mg / self.total_distance_m
                if self.total_distance_m else 0.0
            ),
            "oscillation_switches_per_hr": (
                self.switches * 3600.0 / self.duration_s if self.duration_s else 0.0
            ),
            "duration_s": self.duration_s,
            "steps": self.steps,
            "arrived_vehicles": self.throughput,
            "phase_switches": self.switches,
        }


def _population_variance(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mean = fmean(values)
    return fmean((value - mean) ** 2 for value in values)


def run_seed_sweep(
    evaluate_seed: Callable[[int], Mapping[str, float]],
    *,
    seeds: Sequence[int] | None = None,
    seed_start: int = 1000,
    num_seeds: int = 50,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    """Run an evaluation callback on 50 (by default) reproducible SUMO seeds.

    ``evaluate_seed`` owns the rollout and should use this module's logger to
    collect one episode. It receives the seed and returns the logger summary.
    The function returns per-seed rows and population mean/std for each metric.
    """
    selected_seeds = list(seeds) if seeds is not None else list(
        range(int(seed_start), int(seed_start) + int(num_seeds))
    )
    if not selected_seeds:
        raise ValueError("At least one seed is required.")
    if len(set(selected_seeds)) != len(selected_seeds):
        raise ValueError("Seed list must not contain duplicates.")

    meta = dict(metadata or {})
    required_metadata = {"model_name", "network", "demand_regime"}
    missing_metadata = sorted(required_metadata - set(meta))
    if missing_metadata:
        raise ValueError(
            "metadata must include model_name, network, and demand_regime "
            f"to match EPISODE_SUMMARY_COLUMNS; missing {missing_metadata}"
        )
    rows: list[dict[str, Any]] = []
    for seed in selected_seeds:
        result = dict(evaluate_seed(int(seed)))
        missing = [column for column in METRIC_COLUMNS if column not in result]
        if missing:
            raise ValueError(f"Seed {seed} result is missing metrics: {missing}")
        row = {**result, **meta, "seed": int(seed)}
        missing_schema = [column for column in EPISODE_SUMMARY_COLUMNS if column not in row]
        if missing_schema:
            raise ValueError(
                f"Seed {seed} result does not satisfy EPISODE_SUMMARY_COLUMNS: "
                f"{missing_schema}"
            )
        rows.append(row)

    aggregates: dict[str, dict[str, float]] = {}
    for column in METRIC_COLUMNS:
        values = [float(row[column]) for row in rows]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Metric {column!r} contains a non-finite value.")
        aggregates[column] = {"mean": fmean(values), "std": pstdev(values)}
    return rows, aggregates


def write_seed_sweep_csv(
    rows: Sequence[Mapping[str, Any]],
    aggregates: Mapping[str, Mapping[str, float]],
    output_path: str | Path,
) -> tuple[Path, Path]:
    """Write per-seed metrics and a companion mean/std CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows_path = path
    aggregate_path = path.with_name(f"{path.stem}_summary{path.suffix or '.csv'}")
    if rows:
        fields = list(dict.fromkeys(key for row in rows for key in row.keys()))
        with rows_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    with aggregate_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("metric", "mean", "std"))
        for metric, values in aggregates.items():
            writer.writerow((metric, values["mean"], values["std"]))
    return rows_path, aggregate_path
