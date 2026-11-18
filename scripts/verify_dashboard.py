from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from dashboard.data_loader import (
    LiveRolloutSession,
    compute_view_kpis,
    discover_recorded_logs,
    load_recorded_log,
    slice_episode_view,
)
from scripts.run_rollout import run_rollout
from src.baselines.vanilla_dqn import VanillaDQNAgent
from src.utils.contracts import (
    EPISODE_SUMMARY_COLUMNS,
    OBS_DIM,
    TELEMETRY_COLUMNS,
    load_default_config,
)
from src.utils.controllers import (
    build_controller,
    discover_checkpoints,
    get_controller_availabilities,
    make_traffic_env,
    register_d3qn_loader,
)
from src.utils.telemetry import reconstruct_phase_audit, validate_telemetry_columns


def create_sample_vanilla_dqn_checkpoint(ckpt_path: Path, seed: int = 42) -> Path:
    """Train a VanillaDQNAgent briefly on SUMO and save a valid .pt checkpoint."""
    cfg = load_default_config()
    env = make_traffic_env(
        cfg=cfg,
        net_file="networks/single_intersection/single-intersection.net.xml",
        route_file="networks/single_intersection/single-intersection-vhvh.rou.xml",
        use_shield=False,
        reward_mode="queue_diff",
        num_seconds=200,
        delta_time_s=5,
        validate_contracts=True,
    )
    try:
        agent = VanillaDQNAgent(
            state_dim=OBS_DIM,
            num_actions=env.action_space.n,
            batch_size=16,
            eps_decay_steps=40,
        )
        obs, _ = env.reset(seed=seed)
        done = False
        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            agent.store_transition(obs, action, reward, next_obs, float(done))
            agent.train_step()
            obs = next_obs
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        agent.save(str(ckpt_path))
    finally:
        env.close()
    return ckpt_path


def verify_recorded_fixed_time_log() -> None:
    logs, invalid = discover_recorded_logs("outputs")
    assert not invalid, f"Unexpected invalid telemetry files in outputs/: {invalid}"
    fixed_entries = [e for e in logs if "fixed_time" in e.run_name]
    assert fixed_entries, "Fixed-Time 3600s baseline log not found in outputs/."

    entry = fixed_entries[0]
    view = load_recorded_log(entry)
    assert list(view.telemetry_df.columns) == TELEMETRY_COLUMNS
    validate_telemetry_columns(view.telemetry_df)
    assert len(view.telemetry_df) == 720, f"Expected 720 steps for 3600s, got {len(view.telemetry_df)}"
    assert float(view.telemetry_df["sim_time"].max()) == 3600.0
    assert view.lane_df is not None and len(view.lane_df) == 720 * 8
    assert view.lane_df["lane_id"].nunique() == 8

    summary_df = pd.read_csv(entry.summary_csv_path)
    assert list(summary_df.columns) == EPISODE_SUMMARY_COLUMNS

    kpis = compute_view_kpis(view)
    for k in (
        "avg_vehicle_delay_s_per_veh",
        "max_lane_wait_s",
        "lane_wait_variance_s2",
        "max_queue_vehicles",
        "throughput_veh_per_hr",
        "total_co2_g_per_km",
        "oscillation_switches_per_hr",
    ):
        assert kpis[k] is not None and np.isfinite(kpis[k]), f"Invalid KPI {k}: {kpis[k]}"

    sliced = slice_episode_view(view, max_sim_time=600.0)
    assert len(sliced.telemetry_df) == 120
    assert float(sliced.telemetry_df["sim_time"].max()) == 600.0

    audit = reconstruct_phase_audit(view.telemetry_df)
    assert audit.total_steps == 720
    assert audit.total_switches == 102
    assert audit.shield_interventions == 0
    assert audit.min_green_violations == 0
    yellow_count = int((audit.signal_intervals_df["signal_state"] == "Yellow (3s)").sum())
    red_count = int((audit.signal_intervals_df["signal_state"] == "All-Red (2s)").sum())
    assert yellow_count == 102 and red_count == 102
    print("Fixed-Time 3600s log verification passed.")


def verify_parquet_and_d3qn_hooks() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        res = run_rollout(
            controller="max_pressure",
            seed=42,
            num_seconds=120,
            output_dir=tmpdir,
            file_format="parquet",
        )
        assert res.num_steps == 24
        assert "telemetry_parquet" in res.exported_files
        logs, invalid = discover_recorded_logs(tmpdir)
        assert not invalid and len(logs) == 1
        view = load_recorded_log(logs[0])
        assert len(view.telemetry_df) == 24
        assert view.lane_df is not None and len(view.lane_df) == 24 * 8

    avail = get_controller_availabilities()
    assert avail["fixed_time"].ready and avail["max_pressure"].ready and avail["vanilla_dqn"].ready
    for d3qn_key in ("d3qn_a", "d3qn_b", "d3qn_c", "d3qn_d"):
        assert not avail[d3qn_key].code_available
        assert not avail[d3qn_key].ready

    class _MockD3QNAdapter:
        name = "MockD3QN"

        def reset(self) -> None:
            pass

        def select_action(self, obs, info, env) -> int:
            return int(info["active_phase"])

    try:
        register_d3qn_loader(lambda ckpt, num_actions, cfg, spec: _MockD3QNAdapter())
        hooked_avail = get_controller_availabilities()
        assert hooked_avail["d3qn_d"].code_available
        ctrl = build_controller("d3qn_d", num_actions=4, checkpoint_path="dummy.pt")
        assert ctrl.name == "MockD3QN"
    finally:
        register_d3qn_loader(None)
    print("Parquet export & D3QN availability/hook verification passed.")


def verify_live_vanilla_dqn_session(ckpt_path: Path) -> None:
    discovered = discover_checkpoints()
    assert any(p.resolve() == ckpt_path.resolve() for p in discovered), (
        f"Checkpoint {ckpt_path} not discovered in {discovered}"
    )

    session = LiveRolloutSession()
    session.start(
        controller_key="vanilla_dqn",
        net_file="networks/single_intersection/single-intersection.net.xml",
        route_file="networks/single_intersection/single-intersection-vhvh.rou.xml",
        seed=42,
        num_seconds=120,
        checkpoint_path=ckpt_path,
    )
    assert session.use_shield is False
    assert session.reward_mode == "queue_diff"
    try:
        ticks = 0
        while not session.done:
            taken = session.step_n(4)
            assert taken > 0
            ticks += 1
            live_view = session.to_episode_view()
            assert list(live_view.telemetry_df.columns) == TELEMETRY_COLUMNS
        assert len(session.recorder) == 24
        assert session.env is None
    finally:
        session.close_env()
    print(f"LiveRolloutSession with VanillaDQN checkpoint passed ({ticks} live ticks).")


def verify_streamlit_apptest(ckpt_path: Path) -> None:
    at = AppTest.from_file(str(REPO_ROOT / "dashboard/app.py"), default_timeout=30)
    at.run()
    assert not at.exception, f"AppTest Recorded Mode raised: {at.exception}"
    metric_map = {m.label: m.value for m in at.metric}
    for expected in (
        "Avg Vehicle Delay (s/veh)",
        "Max Lane Wait (s)",
        "Lane Wait Variance (s²)",
        "Max Intersection Queue (vehicles)",
        "Intersection Throughput (veh/hr)",
        "Total CO₂ Emissions (g/km)",
        "Action Oscillation Freq (switches/hr)",
    ):
        assert expected in metric_map, f"Missing KPI metric '{expected}' in {list(metric_map)}"

    # Verify Altair / Vega-Lite charts include 8-lane data, 60s starvation rule, 3s/2s clearances, and 10s min_green.
    import pyarrow as pa

    charts = [e for e in at if "vega" in getattr(e, "type", "")]
    assert len(charts) >= 5, f"Expected >= 5 Altair charts, got {len(charts)} (types={sorted({e.type for e in at})})"

    chart_tables: list[pd.DataFrame] = []
    for c in charts:
        if c.proto.data.data:
            chart_tables.append(pa.ipc.open_stream(c.proto.data.data).read_all().to_pandas())
        for ds in c.proto.datasets:
            if ds.data.data:
                chart_tables.append(pa.ipc.open_stream(ds.data.data).read_all().to_pandas())

    all_lanes_rendered = set()
    has_60s_marker = False
    has_10s_min_green = False
    has_yellow_3s = False
    has_all_red_2s = False
    for tbl in chart_tables:
        if "lane_id" in tbl.columns:
            all_lanes_rendered.update(tbl["lane_id"].unique().tolist())
        if "threshold" in tbl.columns and "label" in tbl.columns:
            if (tbl["threshold"] == 60.0).any() and tbl["label"].astype(str).str.contains("60s").any():
                has_60s_marker = True
        if "min_green" in tbl.columns and (tbl["min_green"] == 10.0).any():
            has_10s_min_green = True
        if "signal_state" in tbl.columns:
            states = set(tbl["signal_state"].astype(str).tolist())
            if "Yellow (3s)" in states:
                has_yellow_3s = True
            if "All-Red (2s)" in states:
                has_all_red_2s = True

    assert len(all_lanes_rendered) == 8, f"Expected 8 lanes in charts, got {all_lanes_rendered}"
    assert has_60s_marker, "Missing W_thresh = 60s starvation threshold marker in charts."
    assert has_10s_min_green, "Missing t_min_green = 10s constraint marker in charts."
    assert has_yellow_3s and has_all_red_2s, "Missing 3s Yellow / 2s All-Red clearance transitions in timeline."

    # Verify Step Telemetry table matches TELEMETRY_COLUMNS and download button is present.
    telemetry_tables = [df for df in at.dataframe if list(df.value.columns) == TELEMETRY_COLUMNS]
    assert telemetry_tables, "Step Telemetry dataframe with exact TELEMETRY_COLUMNS not found."
    assert len(telemetry_tables[0].value) == 720

    # Verify starvation >= 60s filter on Step Telemetry Table works cleanly.
    starv_cb = next(cb for cb in at.checkbox if "Starvation" in cb.label)
    starv_cb.check().run()
    assert not at.exception
    filtered_tables = [df for df in at.dataframe if list(df.value.columns) == TELEMETRY_COLUMNS]
    assert len(filtered_tables[0].value) < 720
    assert (filtered_tables[0].value["max_lane_wait_s"] >= 60.0).all()
    starv_cb.uncheck().run()

    # Verify simulation time scrubber works cleanly.
    scrub_sliders = [s for s in at.sidebar.slider if "Replay Cursor" in s.label]
    assert scrub_sliders, "Replay Cursor slider missing in sidebar."
    scrub_sliders[0].set_value(600.0).run()
    assert not at.exception, f"AppTest after scrubbing raised: {at.exception}"

    # Switch to Live Checkpoint Rollout mode and step VanillaDQN checkpoint end-to-end.
    at.sidebar.radio[0].set_value("Live Checkpoint Rollout").run()
    assert not at.exception, f"Switching to Live mode raised: {at.exception}"

    ctrl_box = next(sb for sb in at.sidebar.selectbox if sb.label == "Controller / Agent")
    ctrl_box.set_value("vanilla_dqn").run()
    assert not at.exception

    ckpt_box = next(sb for sb in at.sidebar.selectbox if sb.label == "PyTorch Checkpoint (.pt)")
    matching_opt = next((opt for opt in ckpt_box.options if ckpt_path.name in opt), None)
    assert matching_opt is not None, f"Checkpoint {ckpt_path.name} missing from {ckpt_box.options}"
    ckpt_box.set_value(matching_opt).run()

    dur_box = next(sb for sb in at.sidebar.selectbox if sb.label == "Duration (s)")
    dur_box.set_value(120).run()

    tick_slider = next(s for s in at.sidebar.slider if s.label == "Steps per UI tick")
    tick_slider.set_value(8).run()

    start_btn = next(b for b in at.sidebar.button if "Start" in b.label)
    start_btn.click().run()
    assert not at.exception, f"Live rollout start raised: {at.exception}"

    live_session: LiveRolloutSession = at.session_state["live_session"]
    try:
        assert len(live_session.recorder) > 0, "Expected live fragment tick to record steps."
        # Trigger fragment reruns until the 120s (24-step) episode finishes.
        for _ in range(10):
            if live_session.done:
                break
            at.run()
            assert not at.exception, f"Live fragment rerun raised: {at.exception}"
        assert live_session.done, "Expected Live VanillaDQN rollout to complete across fragment ticks."
        assert len(live_session.recorder) == 24
        assert live_session.env is None
        live_metrics = {m.label for m in at.metric}
        assert "Avg Vehicle Delay (s/veh)" in live_metrics
        assert "Shield Blocks (mask_approved=False)" in live_metrics
    finally:
        live_session.close_env()
    print("Streamlit AppTest (Recorded Mode + Scrubbing + Live VanillaDQN Fragment Mode) passed.")


def main() -> None:
    verify_recorded_fixed_time_log()
    verify_parquet_and_d3qn_hooks()
    ckpt_path = create_sample_vanilla_dqn_checkpoint(Path("outputs/checkpoints/vanilla_dqn_demo.pt"))
    verify_live_vanilla_dqn_session(ckpt_path)
    verify_streamlit_apptest(ckpt_path)
    print("All dashboard & telemetry verifications passed!")


if __name__ == "__main__":
    main()
