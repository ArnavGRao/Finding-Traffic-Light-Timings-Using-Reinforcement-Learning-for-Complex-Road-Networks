from __future__ import annotations

import csv
import math
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import sumolib
import traci

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.contracts import EPISODE_SUMMARY_COLUMNS, load_default_config
from src.utils.metrics import (
    METRIC_COLUMNS,
    TraCIMetricsLogger,
    run_seed_sweep,
    write_seed_sweep_csv,
)


def _sum_approach_flows(rou_path: Path, begin: float | None = None, end: float | None = None) -> dict[str, float]:
    tree = ET.parse(rou_path)
    totals = {"n": 0.0, "s": 0.0, "e": 0.0, "w": 0.0}
    for flow in tree.getroot().findall("flow"):
        if begin is not None and float(flow.attrib["begin"]) != begin:
            continue
        if end is not None and float(flow.attrib["end"]) != end:
            continue
        approach = flow.attrib["id"][0]
        totals[approach] += float(flow.attrib["vehsPerHour"])
    return totals


def verify_synthetic_demand_files() -> None:
    base = REPO_ROOT / "networks/single_intersection"
    bal = _sum_approach_flows(base / "demand_balanced_symmetric.rou.xml", begin=0, end=3600)
    assert bal == {"n": 600.0, "s": 600.0, "e": 600.0, "w": 600.0}, f"Unexpected balanced flows: {bal}"

    asym = _sum_approach_flows(base / "demand_asymmetric_peak_arterial.rou.xml", begin=0, end=3600)
    assert asym == {"n": 900.0, "s": 900.0, "e": 250.0, "w": 250.0}, f"Unexpected asymmetric flows: {asym}"

    tidal_p1 = _sum_approach_flows(base / "demand_stepwise_tidal.rou.xml", begin=0, end=1200)
    tidal_p2 = _sum_approach_flows(base / "demand_stepwise_tidal.rou.xml", begin=1200, end=2400)
    tidal_p3 = _sum_approach_flows(base / "demand_stepwise_tidal.rou.xml", begin=2400, end=3600)
    assert tidal_p1 == {"n": 600.0, "s": 600.0, "e": 600.0, "w": 600.0}
    assert tidal_p2 == {"n": 900.0, "s": 900.0, "e": 250.0, "w": 250.0}
    assert tidal_p3 == {"n": 250.0, "s": 250.0, "e": 900.0, "w": 900.0}

    sumo_bin = sumolib.checkBinary("sumo")
    net_file = str(base / "single-intersection.net.xml")
    for name in (
        "demand_balanced_symmetric.rou.xml",
        "demand_asymmetric_peak_arterial.rou.xml",
        "demand_stepwise_tidal.rou.xml",
    ):
        traci.start(
            [sumo_bin, "-n", net_file, "-r", str(base / name), "--no-warnings", "--no-step-log", "--seed", "1000"],
            label=f"verify_{name}",
        )
        try:
            logger = TraCIMetricsLogger(traci)
            for _ in range(30):
                traci.simulationStep()
                logger.record_step()
            summary = logger.summary()
        finally:
            traci.close()

        assert summary["steps"] == 30
        assert logger.total_distance_m > 0.0
        expected_g = logger.total_co2_mg / 1_000.0
        assert math.isclose(summary["total_co2_g"], expected_g, rel_tol=1e-9), (
            f"Expected total_co2_g in grams ({expected_g}), got {summary['total_co2_g']}"
        )
        expected_g_per_km = summary["total_co2_g"] / (logger.total_distance_m / 1_000.0)
        assert math.isclose(summary["total_co2_g_per_km"], expected_g_per_km, rel_tol=1e-9)

    print("Synthetic demand regimes & CO2 unit consistency verified.")


def verify_resco_4x4() -> None:
    sumo_bin = sumolib.checkBinary("sumo")
    sumocfg = str(REPO_ROOT / "networks/resco_4x4/grid4x4.sumocfg")
    traci.start([sumo_bin, "-c", sumocfg, "--no-warnings", "--no-step-log", "--seed", "1000"], label="verify_resco")
    try:
        logger = TraCIMetricsLogger(traci)
        for _ in range(60):
            traci.simulationStep()
            logger.record_step()
        summary = logger.summary()
    finally:
        traci.close()

    assert len(logger.traffic_light_ids) == 16, f"Expected 16 TLS in RESCO 4x4, got {len(logger.traffic_light_ids)}"
    total_lanes = sum(len(lanes) for lanes in logger.incoming_lanes.values())
    assert total_lanes == 192, f"Expected 192 incoming lanes in RESCO 4x4, got {total_lanes}"
    assert summary["steps"] == 60
    assert summary["total_co2_g"] > 0.0
    print(f"RESCO 4x4 verified (16 TLS, {total_lanes} lanes, co2_g={summary['total_co2_g']:.2f}).")


def verify_50_seed_sweep() -> None:
    cfg = load_default_config(str(REPO_ROOT / "configs/default.json"))
    num_seeds = int(cfg["evaluation"]["num_seeds"])
    seed_start = int(cfg["evaluation"]["seed_start"])
    assert num_seeds == 50 and seed_start == 1000

    # Sample real TraCI summaries on 2 seeds, and exercise run_seed_sweep across all 50 seeds.
    sumo_bin = sumolib.checkBinary("sumo")
    net_file = str(REPO_ROOT / "networks/single_intersection/single-intersection.net.xml")
    rou_file = str(REPO_ROOT / "networks/single_intersection/demand_balanced_symmetric.rou.xml")
    cached_samples: dict[int, dict[str, float | int]] = {}
    for sample_seed in (seed_start, seed_start + 1):
        traci.start(
            [sumo_bin, "-n", net_file, "-r", rou_file, "--no-warnings", "--no-step-log", "--seed", str(sample_seed)],
            label=f"sweep_sample_{sample_seed}",
        )
        try:
            logger = TraCIMetricsLogger(traci)
            for _ in range(20):
                traci.simulationStep()
                logger.record_step()
            cached_samples[sample_seed] = logger.summary()
        finally:
            traci.close()

    def evaluate_seed(seed: int) -> dict[str, float | int]:
        if seed in cached_samples:
            return cached_samples[seed]
        base = dict(cached_samples[seed_start])
        offset = float(seed - seed_start)
        base["avg_vehicle_delay_s_per_veh"] = float(base["avg_vehicle_delay_s_per_veh"]) + 0.01 * offset
        return base

    rows, aggregates = run_seed_sweep(
        evaluate_seed,
        seed_start=seed_start,
        num_seeds=num_seeds,
        metadata={
            "model_name": "Fixed-Time",
            "network": "single_intersection",
            "demand_regime": "balanced_symmetric",
        },
    )
    assert len(rows) == 50
    assert [r["seed"] for r in rows] == list(range(1000, 1050))
    for col in METRIC_COLUMNS:
        assert col in aggregates and math.isfinite(aggregates[col]["mean"]) and math.isfinite(aggregates[col]["std"])

    with tempfile.TemporaryDirectory() as tmpdir:
        rows_path, agg_path = write_seed_sweep_csv(rows, aggregates, Path(tmpdir) / "sweep_50.csv")
        with rows_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            csv_rows = list(reader)
            assert len(csv_rows) == 50
            for col in EPISODE_SUMMARY_COLUMNS:
                assert col in (reader.fieldnames or []), f"Missing {col} in {reader.fieldnames}"
        with agg_path.open("r", encoding="utf-8") as f:
            agg_rows = list(csv.DictReader(f))
            assert len(agg_rows) == len(METRIC_COLUMNS)

    print("50-seed sweep & CSV export verification passed.")


def main() -> None:
    verify_synthetic_demand_files()
    verify_resco_4x4()
    verify_50_seed_sweep()
    print("All networks & 50-seed TraCI metric checks passed.")


if __name__ == "__main__":
    main()
