from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dashboard.charts import (
    format_lane_label,
    format_phase_label,
    build_comparison_timeseries_chart,
    build_green_duration_chart,
    build_per_lane_timeseries_chart,
    build_proposed_vs_executed_chart,
    build_signal_timeline_chart,
)
from dashboard.data_loader import (
    KPI_METADATA,
    LiveRolloutSession,
    LoadedEpisodeView,
    RecordedLogEntry,
    compute_kpi_deltas,
    compute_view_kpis,
    get_dashboard_environment_status,
    load_recorded_log,
    slice_episode_view,
)
from src.utils.contracts import TELEMETRY_COLUMNS, load_default_config
from src.utils.controllers import CONTROLLER_SPECS
from src.utils.telemetry import reconstruct_phase_audit


st.set_page_config(
    page_title="D3QN Traffic Signal Telemetry & Safety Audit",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def _cached_load_recorded_log(
    telemetry_path_str: str,
    mtime_ns: int,
    lanes_path_str: str | None,
    summary_json_str: str | None,
    summary_csv_str: str | None,
    run_name: str,
    model_name: str,
    network: str,
    demand_regime: str,
    seed: int | None,
    display_label: str,
) -> LoadedEpisodeView:
    """Cache static recorded log reads keyed by file path and modification time."""
    del mtime_ns  # Cache-busting key only.
    entry = RecordedLogEntry(
        run_name=run_name,
        telemetry_path=Path(telemetry_path_str),
        lanes_path=Path(lanes_path_str) if lanes_path_str else None,
        summary_json_path=Path(summary_json_str) if summary_json_str else None,
        summary_csv_path=Path(summary_csv_str) if summary_csv_str else None,
        model_name=model_name,
        network=network,
        demand_regime=demand_regime,
        seed=seed,
        display_label=display_label,
    )
    return load_recorded_log(entry, cfg=load_default_config())


def _load_entry_cached(entry: RecordedLogEntry) -> LoadedEpisodeView:
    stat = entry.telemetry_path.stat()
    return _cached_load_recorded_log(
        telemetry_path_str=str(entry.telemetry_path),
        mtime_ns=stat.st_mtime_ns,
        lanes_path_str=str(entry.lanes_path) if entry.lanes_path else None,
        summary_json_str=str(entry.summary_json_path) if entry.summary_json_path else None,
        summary_csv_str=str(entry.summary_csv_path) if entry.summary_csv_path else None,
        run_name=entry.run_name,
        model_name=entry.model_name,
        network=entry.network,
        demand_regime=entry.demand_regime,
        seed=entry.seed,
        display_label=entry.display_label,
    )


def _get_live_session() -> LiveRolloutSession:
    if "live_session" not in st.session_state:
        st.session_state["live_session"] = LiveRolloutSession()
    return st.session_state["live_session"]


def _format_kpi_value(key: str, val: float | None) -> str:
    if val is None:
        return "N/A"
    if key in {"throughput_veh_per_hr", "oscillation_switches_per_hr", "max_queue_vehicles"}:
        return f"{val:,.1f}"
    if key == "lane_wait_variance_s2":
        return f"{val:,.1f}"
    return f"{val:,.2f}"


def _format_delta_str(key: str, diff: float | None, pct: float | None) -> str | None:
    if diff is None:
        return None
    if abs(diff) < 1e-9:
        return "0.00 (baseline)"
    pct_str = f" ({pct:+.1f}%)" if pct is not None else ""
    if key in {"throughput_veh_per_hr", "oscillation_switches_per_hr", "max_queue_vehicles"}:
        return f"{diff:+.1f}{pct_str}"
    return f"{diff:+.2f}{pct_str}"


def _render_kpi_section(
    active_kpis: dict[str, float | None],
    baseline_kpis: dict[str, float | None] | None,
    baseline_name: str | None,
) -> None:
    st.subheader("Top-Level Performance KPIs")
    if baseline_kpis and baseline_name:
        st.caption(f"Comparison deltas measured against baseline: **{baseline_name}**")
    else:
        st.caption("Select a baseline log in the sidebar to inspect side-by-side comparison deltas.")

    deltas = compute_kpi_deltas(active_kpis, baseline_kpis)

    row1 = st.columns(4)
    row2 = st.columns(3)
    card_slots = row1 + row2

    for slot, (key, title, unit, delta_color) in zip(card_slots, KPI_METADATA):
        val_str = _format_kpi_value(key, active_kpis.get(key))
        delta_info = deltas[key]
        delta_str = _format_delta_str(key, delta_info["diff"], delta_info["pct"])
        with slot:
            st.metric(
                label=f"{title} ({unit})",
                value=val_str,
                delta=delta_str,
                delta_color=delta_color,
            )

    if baseline_kpis and baseline_name:
        with st.expander("Side-by-Side KPI Comparison Table", expanded=False):
            rows = []
            for key, title, unit, _ in KPI_METADATA:
                cur_v = active_kpis.get(key)
                base_v = baseline_kpis.get(key)
                d_info = deltas[key]
                rows.append(
                    {
                        "Metric": f"{title} ({unit})",
                        "Active Run": round(cur_v, 3) if cur_v is not None else None,
                        f"Baseline ({baseline_name})": round(base_v, 3) if base_v is not None else None,
                        "Delta": round(d_info["diff"], 3) if d_info["diff"] is not None else None,
                        "Delta (%)": round(d_info["pct"], 2) if d_info["pct"] is not None else None,
                    }
                )
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _render_queue_and_starvation_section(
    view: LoadedEpisodeView,
    baseline_view: LoadedEpisodeView | None,
) -> None:
    st.subheader("Queue and Starvation Monitoring (8 Incoming Lanes)")
    w_thresh = float(view.metadata.get("w_thresh_s", 60.0))
    t_df = view.telemetry_df
    l_df = view.lane_df
    b_df = baseline_view.telemetry_df if baseline_view is not None else None
    b_label = baseline_view.model_name if baseline_view is not None else "Baseline"

    if l_df is not None and not l_df.empty:
        starvation_hits = l_df[l_df["max_wait_s"] >= w_thresh]
        num_breaches = len(starvation_hits)
        breached_lanes = sorted({format_lane_label(lid) for lid in starvation_hits["lane_id"].unique()})
        peak_wait = float(l_df["max_wait_s"].max())

        b_cols = st.columns(3)
        b_cols[0].metric("Starvation Threshold (W_thresh)", f"{w_thresh:.0f} s")
        b_cols[1].metric("Lane-Step Breaches (≥ 60s)", f"{num_breaches:,}")
        b_cols[2].metric("Peak Observed Wait", f"{peak_wait:.1f} s")
        if breached_lanes:
            st.warning(
                f"Starvation threshold ($W_{{\\text{{thresh}}}} = {w_thresh:.0f}\\text{{s}}$) exceeded on: "
                + ", ".join(breached_lanes)
            )
        else:
            st.success(f"No incoming lane exceeded $W_{{\\text{{thresh}}}} = {w_thresh:.0f}\\text{{s}}$ in this window.")

        col_q, col_w = st.columns(2)
        with col_q:
            st.markdown("**Per-Lane Queue Lengths (Halting Vehicles)**")
            q_chart = build_per_lane_timeseries_chart(
                l_df,
                y_col="queue",
                y_title="Halting Queue (vehicles)",
            )
            st.altair_chart(q_chart, width="stretch")

        with col_w:
            st.markdown(f"**Per-Lane Max Waiting Time vs $W_{{\\text{{thresh}}}} = {w_thresh:.0f}\\text{{s}}$**")
            w_chart = build_per_lane_timeseries_chart(
                l_df,
                y_col="max_wait_s",
                y_title="Max Lane Waiting Time (s)",
                threshold_s=w_thresh,
                threshold_label=f"W_thresh = {w_thresh:.0f}s (Starvation)",
            )
            st.altair_chart(w_chart, width="stretch")
    else:
        st.info(
            "Per-lane breakdown (`*_lanes.csv`) was not found alongside this telemetry file; "
            "rendering aggregate intersection queue and starvation charts below."
        )

    with st.expander("Intersection Aggregate & Baseline Overlay Trends", expanded=b_df is not None):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Total Intersection Queue**")
            st.altair_chart(
                build_comparison_timeseries_chart(
                    t_df,
                    y_col="total_queue",
                    y_title="Total Queue (veh)",
                    active_label=view.model_name,
                    baseline_df=b_df,
                    baseline_label=b_label,
                ),
                width="stretch",
            )
        with c2:
            st.markdown("**Max Lane Wait (Starvation Onset)**")
            st.altair_chart(
                build_comparison_timeseries_chart(
                    t_df,
                    y_col="max_lane_wait_s",
                    y_title="Max Lane Wait (s)",
                    active_label=view.model_name,
                    baseline_df=b_df,
                    baseline_label=b_label,
                    threshold_s=w_thresh,
                ),
                width="stretch",
            )
        with c3:
            st.markdown("**Lane Wait Variance (s²)**")
            st.altair_chart(
                build_comparison_timeseries_chart(
                    t_df,
                    y_col="lane_wait_variance_s2",
                    y_title="Wait Variance (s²)",
                    active_label=view.model_name,
                    baseline_df=b_df,
                    baseline_label=b_label,
                ),
                width="stretch",
            )


def _render_phase_and_shield_section(view: LoadedEpisodeView) -> None:
    st.subheader("Signal Phase & Safety Shield Audit")
    delta_time_s = float(view.metadata.get("delta_time_s", 5.0))
    min_green_s = float(view.metadata.get("min_green_s", 10.0))
    yellow_s = float(view.metadata.get("yellow_s", 3.0))
    all_red_s = float(view.metadata.get("all_red_s", 2.0))

    phase_labels = {i: format_phase_label(i) for i in range(8)}
    audit = reconstruct_phase_audit(
        view.telemetry_df,
        delta_time_s=delta_time_s,
        min_green_s=min_green_s,
        yellow_s=yellow_s,
        all_red_s=all_red_s,
        phase_labels=phase_labels,
    )

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Decision Steps", f"{audit.total_steps:,}")
    m2.metric("Phase Switches", f"{audit.total_switches:,}")
    m3.metric("Shield Blocks (mask_approved=False)", f"{audit.shield_interventions:,}")
    m4.metric("Intervention Rate", f"{audit.intervention_rate_pct:.1f}%")
    m5.metric("t_min_green (<10s) Violations", f"{audit.min_green_violations:,}")

    if not audit.signal_intervals_df.empty:
        st.markdown(
            f"**Signal State Timeline** (Green Phases + Highlighted **{yellow_s:.0f}s Yellow** & **{all_red_s:.0f}s All-Red** Clearance Transitions)"
        )
        st.altair_chart(
            build_signal_timeline_chart(audit.signal_intervals_df),
            width="stretch",
        )

    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown(f"**Active Green Phase Durations vs $t_{{\\min\\_green}} = {min_green_s:.0f}\\text{{s}}$**")
        if not audit.green_blocks_df.empty:
            st.altair_chart(
                build_green_duration_chart(audit.green_blocks_df, min_green_s=min_green_s),
                width="stretch",
            )
        else:
            st.info("No completed green phase blocks in the selected window.")

    with c_right:
        st.markdown("**Proposed vs Executed Actions & Shield Interventions**")
        if not audit.action_audit_df.empty:
            st.altair_chart(
                build_proposed_vs_executed_chart(audit.action_audit_df),
                width="stretch",
            )

    tab_blocks, tab_interventions, tab_switches = st.tabs(
        [
            f"Green Phase Blocks ({len(audit.green_blocks_df)})",
            f"Shield Interventions ({audit.shield_interventions})",
            f"Clearance Transitions ({audit.total_switches})",
        ]
    )
    with tab_blocks:
        if not audit.green_blocks_df.empty:
            st.dataframe(audit.green_blocks_df, width="stretch", hide_index=True)
        else:
            st.caption("No green blocks recorded yet.")
    with tab_interventions:
        blocked_df = audit.action_audit_df[audit.action_audit_df["shield_intervened"]]
        if not blocked_df.empty:
            st.dataframe(blocked_df, width="stretch", hide_index=True)
        else:
            st.caption("Zero safety-shield interventions (`mask_approved == False`) in this window.")
    with tab_switches:
        switch_df = audit.action_audit_df[audit.action_audit_df["phase_switched"]]
        if not switch_df.empty:
            st.dataframe(switch_df, width="stretch", hide_index=True)
        else:
            st.caption("No phase transitions in this window.")


def _render_telemetry_table_section(view: LoadedEpisodeView, is_live_running: bool) -> None:
    st.subheader("Step Telemetry Table & Export")
    df = view.telemetry_df[list(TELEMETRY_COLUMNS)].copy()
    if df.empty:
        st.info("No step telemetry records available yet.")
        return

    f1, f2, f3, f4 = st.columns(4)
    phases = sorted(df["active_phase"].unique().tolist())
    with f1:
        selected_phases = st.multiselect(
            "Filter Active Phase",
            options=phases,
            default=phases,
            key=f"tbl_phases_{view.source_mode}",
        )
    with f2:
        mask_filter = st.selectbox(
            "Mask Approved",
            options=["All", "Approved (True)", "Blocked (False)"],
            key=f"tbl_mask_{view.source_mode}",
        )
    with f3:
        switch_filter = st.selectbox(
            "Phase Switched",
            options=["All", "Switched (True)", "Held (False)"],
            key=f"tbl_switch_{view.source_mode}",
        )
    with f4:
        starvation_only = st.checkbox(
            "Starvation ≥ 60s only",
            value=False,
            key=f"tbl_starv_{view.source_mode}",
        )

    filtered = df[df["active_phase"].isin(selected_phases)] if selected_phases else df.iloc[0:0]
    min_t, max_t = float(df["sim_time"].min()), float(df["sim_time"].max())
    if not is_live_running and max_t > min_t:
        t_start, t_end = st.slider(
            "Filter Simulation Time Window (s)",
            min_value=min_t,
            max_value=max_t,
            value=(min_t, max_t),
            step=float(view.metadata.get("delta_time_s", 5.0)),
            key=f"tbl_time_{view.source_mode}_{view.run_name}_{len(df)}",
        )
        filtered = filtered[(filtered["sim_time"] >= t_start) & (filtered["sim_time"] <= t_end)]
    if mask_filter == "Approved (True)":
        filtered = filtered[filtered["mask_approved"]]
    elif mask_filter == "Blocked (False)":
        filtered = filtered[~filtered["mask_approved"]]

    if switch_filter == "Switched (True)":
        filtered = filtered[filtered["phase_switched"]]
    elif switch_filter == "Held (False)":
        filtered = filtered[~filtered["phase_switched"]]

    if starvation_only:
        w_thresh = float(view.metadata.get("w_thresh_s", 60.0))
        filtered = filtered[filtered["max_lane_wait_s"] >= w_thresh]

    st.caption(f"Showing **{len(filtered):,}** of **{len(df):,}** step records (`TELEMETRY_COLUMNS`).")
    st.dataframe(filtered, width="stretch", hide_index=True, height=320)

    if not is_live_running:
        csv_bytes = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Filtered Telemetry CSV",
            data=csv_bytes,
            file_name=f"{view.run_name}_telemetry_filtered.csv",
            mime="text/csv",
            key=f"dl_btn_{view.source_mode}",
        )


def _render_dashboard_body(
    view: LoadedEpisodeView,
    baseline_view: LoadedEpisodeView | None,
    scrub_max_t: float | None = None,
    is_live_running: bool = False,
) -> None:
    if view.telemetry_df.empty:
        st.info("Waiting for initial simulation steps...")
        return

    active_view = slice_episode_view(view, max_sim_time=scrub_max_t) if scrub_max_t is not None else view
    scoped_baseline = baseline_view
    if baseline_view is not None and scrub_max_t is not None:
        base_max = float(baseline_view.telemetry_df["sim_time"].max())
        if scrub_max_t < base_max:
            scoped_baseline = slice_episode_view(baseline_view, max_sim_time=scrub_max_t)

    t_df = active_view.telemetry_df
    latest = t_df.iloc[-1]

    with st.container(border=True):
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.markdown(f"**Model**\n\n`{active_view.model_name}`")
        c2.markdown(f"**Network / Demand**\n\n`{active_view.network}` / `{active_view.demand_regime}` (seed `{active_view.seed}`)")
        c3.markdown(f"**Sim Time**\n\n`{float(latest['sim_time']):.1f} s`")
        c4.markdown(f"**Active Phase**\n\n`{format_phase_label(int(latest['active_phase']))}` (`{float(latest['elapsed_green_s']):.0f}s`)")
        c5.markdown(
            f"**Action (Prop → Exec)**\n\n`{int(latest['proposed_action'])}` → `{int(latest['executed_action'])}` "
            + ("✅" if bool(latest["mask_approved"]) else "🛡️ Blocked")
        )
        c6.markdown(
            f"**Queue / Max Wait**\n\n`{float(latest['total_queue']):.0f} veh` / `{float(latest['max_lane_wait_s']):.1f} s`"
        )

    active_kpis = compute_view_kpis(active_view)
    baseline_kpis = compute_view_kpis(scoped_baseline) if scoped_baseline is not None else None
    baseline_label = scoped_baseline.model_name if scoped_baseline is not None else None

    chart_baseline = (
        scoped_baseline
        if scoped_baseline is not None
        and not (
            scoped_baseline.run_name == active_view.run_name
            and active_view.source_mode == "recorded"
        )
        else None
    )

    _render_kpi_section(active_kpis, baseline_kpis, baseline_label)
    st.divider()
    _render_queue_and_starvation_section(active_view, chart_baseline)
    st.divider()
    _render_phase_and_shield_section(active_view)
    st.divider()
    _render_telemetry_table_section(active_view, is_live_running=is_live_running)


@st.fragment(run_every=0.3)
def _render_live_streaming_fragment(
    steps_per_tick: int,
    baseline_view: LoadedEpisodeView | None,
) -> None:
    session = _get_live_session()
    if session.is_running and not session.done:
        try:
            session.step_n(steps_per_tick)
        except Exception as exc:
            st.error(f"Live SUMO simulation error: {exc}")
            st.rerun()

    progress = (
        min(1.0, float(session.recorder.records[-1].sim_time) / max(1, session.num_seconds))
        if len(session.recorder) > 0
        else 0.0
    )
    st.progress(
        progress,
        text=f"Live Headless SUMO Rollout: {len(session.recorder)} steps | {progress * 100:.0f}% complete",
    )

    _render_dashboard_body(
        session.to_episode_view(),
        baseline_view=baseline_view,
        scrub_max_t=None,
        is_live_running=session.is_running,
    )

    if session.done or not session.is_running:
        st.rerun()


def main() -> None:
    st.title("🚦 Traffic Signal Control Telemetry & Safety Audit")

    status = get_dashboard_environment_status()
    recorded_logs: list[RecordedLogEntry] = status["recorded_logs"]
    invalid_logs: list[tuple[Path, str]] = status["invalid_logs"]
    availabilities = status["availabilities"]
    all_checkpoints: list[Path] = status["checkpoints"]
    net_files: list[Path] = status["net_files"]
    route_files: list[Path] = status["route_files"]

    live_session = _get_live_session()

    with st.sidebar:
        st.header("Rollout & Data Controls")
        source_mode = st.radio(
            "Data Source",
            options=["Recorded Log", "Live Checkpoint Rollout"],
            key="sidebar_source_mode",
        )

        if source_mode == "Recorded Log" and live_session.is_running:
            live_session.pause()

        selected_entry: RecordedLogEntry | None = None
        if source_mode == "Recorded Log":
            st.subheader("Recorded Telemetry Logs")
            if st.button("Refresh `outputs/`", width="stretch"):
                st.cache_data.clear()
                st.rerun()

            if not recorded_logs:
                st.warning("No valid telemetry logs found in `outputs/`. Run `scripts/run_rollout.py` first.")
            else:
                labels = [e.display_label for e in recorded_logs]
                chosen_idx = st.selectbox(
                    "Select Telemetry Run",
                    options=list(range(len(recorded_logs))),
                    format_func=lambda idx: labels[idx],
                    key="recorded_log_select",
                )
                selected_entry = recorded_logs[chosen_idx]

            if invalid_logs:
                with st.expander(f"Skipped Invalid Files ({len(invalid_logs)})", expanded=False):
                    for path, err in invalid_logs:
                        st.caption(f"`{path.name}`: {err}")

        else:
            st.subheader("Live Checkpoint Evaluation")
            ctrl_keys = list(CONTROLLER_SPECS.keys())

            def _ctrl_label(k: str) -> str:
                avail = availabilities[k]
                spec = avail.spec
                if not spec.is_d3qn:
                    return f"✅ {spec.display_name}"
                if avail.ready:
                    return f"✅ {spec.display_name}"
                if avail.code_available:
                    return f"🟡 {spec.display_name} (awaiting .pt)"
                return f"⏳ {spec.display_name} (unavailable)"

            chosen_ctrl = st.selectbox(
                "Controller / Agent",
                options=ctrl_keys,
                format_func=_ctrl_label,
                key="live_ctrl_select",
            )
            avail = availabilities[chosen_ctrl]
            spec = avail.spec
            st.caption(avail.reason)

            checkpoint_path_str: str | None = None
            allow_untrained = False
            if spec.requires_checkpoint:
                ckpt_options = [str(p) for p in (avail.checkpoint_files or all_checkpoints)]
                choice_list = ["<Custom path...>"] + ckpt_options
                default_idx = 1 if ckpt_options else 0
                selected_ckpt = st.selectbox(
                    "PyTorch Checkpoint (.pt)",
                    options=choice_list,
                    index=default_idx,
                    key="live_ckpt_select",
                )
                if selected_ckpt == "<Custom path...>":
                    custom_ckpt = st.text_input(
                        "Custom .pt Path",
                        value="",
                        placeholder="outputs/checkpoints/model.pt",
                        key="live_ckpt_custom",
                    )
                    checkpoint_path_str = custom_ckpt.strip() or None
                else:
                    checkpoint_path_str = selected_ckpt

                if chosen_ctrl == "vanilla_dqn" and checkpoint_path_str is None:
                    allow_untrained = st.checkbox(
                        "Allow untrained Vanilla DQN (smoke test)",
                        value=False,
                        key="live_allow_untrained",
                    )

            default_net_idx = next(
                (i for i, p in enumerate(net_files) if "single-intersection.net.xml" in p.name),
                0,
            )
            net_file = (
                st.selectbox(
                    "SUMO Network",
                    options=net_files,
                    index=default_net_idx if net_files else 0,
                    format_func=lambda p: str(p),
                    key="live_net_select",
                )
                if net_files
                else Path("networks/single_intersection/single-intersection.net.xml")
            )

            default_route_idx = next(
                (i for i, p in enumerate(route_files) if "vhvh" in p.name),
                0,
            )
            route_file = (
                st.selectbox(
                    "Demand Route File",
                    options=route_files,
                    index=default_route_idx if route_files else 0,
                    format_func=lambda p: p.name,
                    key="live_route_select",
                )
                if route_files
                else Path("networks/single_intersection/single-intersection-vhvh.rou.xml")
            )

            c_seed, c_dur = st.columns(2)
            with c_seed:
                seed = int(st.number_input("Seed", min_value=0, max_value=100_000, value=42, step=1))
            with c_dur:
                num_seconds = int(
                    st.selectbox(
                        "Duration (s)",
                        options=[120, 300, 600, 1800, 3600],
                        index=1,
                    )
                )

            steps_per_tick = int(
                st.slider("Steps per UI tick", min_value=1, max_value=12, value=3, step=1)
            )

            with st.expander("Environment Shield & Reward Flags", expanded=False):
                use_shield = st.checkbox(
                    "Enable Safety Shield (t_min_green=10s)",
                    value=spec.default_use_shield,
                    key=f"shield_{chosen_ctrl}",
                )
                reward_modes = ["starvation", "delay_diff", "queue_diff"]
                reward_mode = st.selectbox(
                    "Reward Mode",
                    options=reward_modes,
                    index=reward_modes.index(spec.default_reward_mode),
                    key=f"reward_{chosen_ctrl}",
                )

            can_start = avail.code_available and (
                not spec.requires_checkpoint
                or checkpoint_path_str is not None
                or allow_untrained
            )

            b1, b2 = st.columns(2)
            with b1:
                start_clicked = st.button(
                    "▶ Start / Restart",
                    disabled=not can_start,
                    type="primary",
                    width="stretch",
                )
            with b2:
                if live_session.is_running:
                    if st.button("⏸ Pause", width="stretch"):
                        live_session.pause()
                        st.rerun()
                elif live_session.is_paused:
                    if st.button("▶ Resume", width="stretch"):
                        live_session.resume()
                        st.rerun()
                else:
                    st.button("⏸ Pause", disabled=True, width="stretch")

            b3, b4 = st.columns(2)
            with b3:
                if st.button(
                    "⏭ Step +1",
                    disabled=live_session.env is None or live_session.done or live_session.is_running,
                    width="stretch",
                ):
                    live_session.step_n(1)
                    st.rerun()
            with b4:
                if st.button(
                    "⏹ Stop",
                    disabled=live_session.env is None and not live_session.is_running,
                    width="stretch",
                ):
                    live_session.stop()
                    st.rerun()

            if start_clicked:
                try:
                    live_session.start(
                        controller_key=chosen_ctrl,
                        net_file=net_file,
                        route_file=route_file,
                        seed=seed,
                        num_seconds=num_seconds,
                        checkpoint_path=checkpoint_path_str,
                        use_shield=use_shield,
                        reward_mode=reward_mode,
                        allow_untrained=allow_untrained,
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(f"Failed to start live rollout: {exc}")

            if len(live_session.recorder) > 0 and not live_session.is_running:
                if st.button("💾 Save Rollout to `outputs/`", width="stretch"):
                    exported = live_session.export_to_outputs(output_dir="outputs", file_format="csv")
                    st.success(f"Exported to `{exported['telemetry_csv'].name}`")
                    st.cache_data.clear()

        st.divider()
        st.subheader("Baseline Comparison Overlay")
        baseline_entry: RecordedLogEntry | None = None
        if recorded_logs:
            fixed_default = next(
                (i + 1 for i, e in enumerate(recorded_logs) if "fixed" in e.run_name.lower()),
                0,
            )
            baseline_choices = [None] + recorded_logs
            baseline_entry = st.selectbox(
                "Baseline Log",
                options=baseline_choices,
                index=fixed_default,
                format_func=lambda e: "None (no baseline overlay)" if e is None else e.display_label,
                key="baseline_log_select",
            )
        else:
            st.caption("No recorded baseline logs in `outputs/` yet.")

    baseline_view = _load_entry_cached(baseline_entry) if baseline_entry is not None else None

    if source_mode == "Recorded Log":
        if selected_entry is None:
            st.info(
                "No recorded telemetry logs available in `outputs/`. "
                "Generate one with `python3 scripts/run_rollout.py --controller fixed_time --seed 42 --num-seconds 3600` "
                "or switch to **Live Checkpoint Rollout** in the sidebar."
            )
            return

        active_view = _load_entry_cached(selected_entry)
        if baseline_view is not None and baseline_view.run_name == active_view.run_name:
            # Allow inspecting baseline directly without redundant identical curve overlay unless desired.
            pass

        scrub_max_t: float | None = None
        if not active_view.telemetry_df.empty:
            min_t = float(active_view.telemetry_df["sim_time"].min())
            max_t = float(active_view.telemetry_df["sim_time"].max())
            dt = float(active_view.metadata.get("delta_time_s", 5.0))
            if max_t > min_t:
                with st.sidebar:
                    st.subheader("Simulation Time Scrubber")
                    scrub_max_t = float(
                        st.slider(
                            "Replay Cursor (sim_time ≤ T)",
                            min_value=min_t,
                            max_value=max_t,
                            value=max_t,
                            step=dt,
                            key=f"scrub_recorded_{active_view.run_name}",
                        )
                    )
                    if scrub_max_t < max_t:
                        st.caption(f"Replaying `0s → {scrub_max_t:.0f}s` of `{max_t:.0f}s`")

        _render_dashboard_body(
            active_view,
            baseline_view=baseline_view,
            scrub_max_t=scrub_max_t,
            is_live_running=False,
        )
    else:
        if live_session.error_message:
            st.error(f"Live Rollout Error: {live_session.error_message}")

        if live_session.is_running:
            _render_live_streaming_fragment(
                steps_per_tick=steps_per_tick,
                baseline_view=baseline_view,
            )
        elif len(live_session.recorder) > 0:
            live_view = live_session.to_episode_view()
            scrub_live_t: float | None = None
            min_t = float(live_view.telemetry_df["sim_time"].min())
            max_t = float(live_view.telemetry_df["sim_time"].max())
            dt = float(live_view.metadata.get("delta_time_s", 5.0))
            if max_t > min_t:
                with st.sidebar:
                    st.subheader("Simulation Time Scrubber")
                    scrub_live_t = float(
                        st.slider(
                            "Replay Cursor (sim_time ≤ T)",
                            min_value=min_t,
                            max_value=max_t,
                            value=max_t,
                            step=dt,
                            key=f"scrub_live_{live_view.run_name}_{len(live_view.telemetry_df)}",
                        )
                    )
            status_word = "Completed" if live_session.done else "Paused"
            st.success(
                f"Live Rollout {status_word}: `{live_view.model_name}` | "
                f"{len(live_view.telemetry_df)} steps recorded (`sim_time = {max_t:.1f}s`)."
            )
            _render_dashboard_body(
                live_view,
                baseline_view=baseline_view,
                scrub_max_t=scrub_live_t,
                is_live_running=False,
            )
        else:
            st.info(
                "Configure a controller, optional `.pt` checkpoint, and simulation seed in the sidebar, "
                "then click **▶ Start / Restart** to stream live headless SUMO telemetry."
            )
            with st.expander("Controller & D3QN Availability Status", expanded=True):
                rows = []
                for key, av in availabilities.items():
                    rows.append(
                        {
                            "Controller": av.spec.display_name,
                            "Key": key,
                            "Shield Default": av.spec.default_use_shield,
                            "Reward Mode": av.spec.default_reward_mode,
                            "Code Available": av.code_available,
                            "Discovered Checkpoints": len(av.checkpoint_files),
                            "Status": av.reason,
                        }
                    )
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
