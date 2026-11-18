from __future__ import annotations

import altair as alt
import pandas as pd


LANE_LABELS: dict[str, str] = {
    "n_t_0": "n_t_0 (N Thru/Right)",
    "n_t_1": "n_t_1 (N Left)",
    "e_t_0": "e_t_0 (E Thru/Right)",
    "e_t_1": "e_t_1 (E Left)",
    "s_t_0": "s_t_0 (S Thru/Right)",
    "s_t_1": "s_t_1 (S Left)",
    "w_t_0": "w_t_0 (W Thru/Right)",
    "w_t_1": "w_t_1 (W Left)",
}

PHASE_LABELS: dict[int, str] = {
    0: "Phase 0 (NS Thru/Right)",
    1: "Phase 1 (NS Left)",
    2: "Phase 2 (EW Thru/Right)",
    3: "Phase 3 (EW Left)",
}


def format_lane_label(lane_id: str) -> str:
    return LANE_LABELS.get(str(lane_id), str(lane_id))


def format_phase_label(phase_id: int) -> str:
    return PHASE_LABELS.get(int(phase_id), f"Phase {int(phase_id)}")


def build_per_lane_timeseries_chart(
    lane_df: pd.DataFrame,
    y_col: str,
    y_title: str,
    threshold_s: float | None = None,
    threshold_label: str | None = None,
    height: int = 330,
) -> alt.Chart:
    """Build an interactive 8-lane Altair time-series chart with optional threshold marker."""
    plot_df = lane_df.copy()
    plot_df["lane_label"] = plot_df["lane_id"].map(format_lane_label)

    lane_select = alt.selection_point(fields=["lane_label"], bind="legend")

    base_chart = (
        alt.Chart(plot_df)
        .mark_line(strokeWidth=2.0)
        .encode(
            x=alt.X("sim_time:Q", title="Simulation Time (s)"),
            y=alt.Y(f"{y_col}:Q", title=y_title),
            color=alt.Color("lane_label:N", title="Incoming Lane"),
            opacity=alt.condition(lane_select, alt.value(0.95), alt.value(0.15)),
            tooltip=[
                alt.Tooltip("sim_time:Q", title="Sim Time (s)", format=".1f"),
                alt.Tooltip("lane_label:N", title="Lane"),
                alt.Tooltip(f"{y_col}:Q", title=y_title, format=".2f"),
                alt.Tooltip("vehicle_count:Q", title="Vehicles on Lane", format=".0f"),
                alt.Tooltip("delay_s:Q", title="Lane Avg Delay (s)", format=".2f"),
            ],
        )
        .add_params(lane_select)
        .properties(height=height)
    )

    if threshold_s is None:
        return base_chart.interactive()

    max_t = float(plot_df["sim_time"].max()) if not plot_df.empty else 0.0
    rule_df = pd.DataFrame(
        [
            {
                "threshold": float(threshold_s),
                "label": threshold_label or f"Threshold ({threshold_s:.0f}s)",
                "x_pos": max_t,
            }
        ]
    )
    rule = (
        alt.Chart(rule_df)
        .mark_rule(color="#ef4444", strokeDash=[6, 4], strokeWidth=2.0)
        .encode(y="threshold:Q")
    )
    label = (
        alt.Chart(rule_df)
        .mark_text(align="right", dx=-6, dy=-8, color="#ef4444", fontWeight="bold")
        .encode(
            x=alt.X("x_pos:Q"),
            y="threshold:Q",
            text="label:N",
        )
    )
    return alt.layer(base_chart, rule, label).properties(height=height)


def build_comparison_timeseries_chart(
    telemetry_df: pd.DataFrame,
    y_col: str,
    y_title: str,
    active_label: str,
    baseline_df: pd.DataFrame | None = None,
    baseline_label: str = "Fixed-Time Baseline",
    threshold_s: float | None = None,
    threshold_label: str | None = None,
    height: int = 270,
) -> alt.Chart:
    """Render aggregate metric comparison between the active run and optional baseline overlay."""
    active_slice = telemetry_df[["sim_time", y_col]].copy()
    active_slice["series"] = active_label
    active_slice["is_baseline"] = False

    frames = [active_slice]
    if baseline_df is not None and not baseline_df.empty and y_col in baseline_df.columns:
        base_slice = baseline_df[["sim_time", y_col]].copy()
        base_slice["series"] = f"Baseline: {baseline_label}"
        base_slice["is_baseline"] = True
        frames.append(base_slice)

    combined = pd.concat(frames, ignore_index=True)

    chart = (
        alt.Chart(combined)
        .mark_line(strokeWidth=2.2)
        .encode(
            x=alt.X("sim_time:Q", title="Simulation Time (s)"),
            y=alt.Y(f"{y_col}:Q", title=y_title),
            color=alt.Color(
                "series:N",
                title="Run",
                scale=alt.Scale(range=["#2563eb", "#64748b"]),
            ),
            strokeDash=alt.StrokeDash(
                "is_baseline:N",
                legend=None,
                scale=alt.Scale(domain=[False, True], range=[[1, 0], [5, 4]]),
            ),
            tooltip=[
                alt.Tooltip("sim_time:Q", title="Sim Time (s)", format=".1f"),
                alt.Tooltip("series:N", title="Series"),
                alt.Tooltip(f"{y_col}:Q", title=y_title, format=".2f"),
            ],
        )
        .properties(height=height)
    )

    if threshold_s is None:
        return chart.interactive()

    rule_df = pd.DataFrame(
        [{"threshold": float(threshold_s), "label": threshold_label or f"W_thresh = {threshold_s:.0f}s"}]
    )
    rule = (
        alt.Chart(rule_df)
        .mark_rule(color="#ef4444", strokeDash=[6, 4], strokeWidth=2.0)
        .encode(y="threshold:Q")
    )
    return alt.layer(chart, rule).properties(height=height)


def _coalesce_green_intervals(intervals_df: pd.DataFrame) -> pd.DataFrame:
    """Merge contiguous Green intervals sharing the same phase for crisp Gantt rendering."""
    if intervals_df.empty:
        return intervals_df
    merged: list[dict] = []
    for row in intervals_df.to_dict("records"):
        if (
            merged
            and row["signal_state"] == "Green"
            and merged[-1]["signal_state"] == "Green"
            and merged[-1]["active_phase"] == row["active_phase"]
            and abs(float(merged[-1]["end_s"]) - float(row["start_s"])) < 1e-6
        ):
            merged[-1]["end_s"] = row["end_s"]
            merged[-1]["duration_s"] = float(merged[-1]["end_s"]) - float(merged[-1]["start_s"])
        else:
            merged.append(dict(row))
    return pd.DataFrame(merged)


def build_signal_timeline_chart(intervals_df: pd.DataFrame, height: int = 210) -> alt.Chart:
    """Visualize Green phases alongside highlighted 3s Yellow and 2s All-Red clearances."""
    plot_df = _coalesce_green_intervals(intervals_df)

    def _state_group(row: pd.Series) -> str:
        if row["signal_state"] == "Green":
            return format_phase_label(int(row["active_phase"]))
        return str(row["signal_state"])

    plot_df["display_state"] = plot_df.apply(_state_group, axis=1)
    plot_df["track"] = plot_df["signal_state"].apply(
        lambda s: "Clearance (Yellow / All-Red)" if s != "Green" else "Active Green Phase"
    )

    color_domain = [
        "Phase 0 (NS Thru/Right)",
        "Phase 1 (NS Left)",
        "Phase 2 (EW Thru/Right)",
        "Phase 3 (EW Left)",
        "Yellow (3s)",
        "All-Red (2s)",
    ]
    color_range = [
        "#10b981",
        "#06b6d4",
        "#3b82f6",
        "#6366f1",
        "#f59e0b",
        "#ef4444",
    ]
    present_states = [s for s in plot_df["display_state"].unique() if s not in color_domain]
    for extra in sorted(present_states):
        color_domain.append(extra)
        color_range.append("#8b5cf6")

    return (
        alt.Chart(plot_df)
        .mark_bar(height=28)
        .encode(
            x=alt.X("start_s:Q", title="Simulation Time (s)"),
            x2="end_s:Q",
            y=alt.Y(
                "track:N",
                title="",
                sort=["Active Green Phase", "Clearance (Yellow / All-Red)"],
            ),
            color=alt.Color(
                "display_state:N",
                title="Signal State",
                scale=alt.Scale(domain=color_domain, range=color_range),
            ),
            tooltip=[
                alt.Tooltip("display_state:N", title="State"),
                alt.Tooltip("phase_label:N", title="Detail"),
                alt.Tooltip("start_s:Q", title="Start (s)", format=".1f"),
                alt.Tooltip("end_s:Q", title="End (s)", format=".1f"),
                alt.Tooltip("duration_s:Q", title="Duration (s)", format=".1f"),
            ],
        )
        .properties(height=height)
        .interactive()
    )


def build_green_duration_chart(
    green_blocks_df: pd.DataFrame,
    min_green_s: float = 10.0,
    height: int = 270,
) -> alt.Chart:
    """Plot green phase block durations against the t_min_green = 10s safety constraint."""
    plot_df = green_blocks_df.copy()

    def _compliance_status(row: pd.Series) -> str:
        if bool(row["violates_min_green"]):
            return f"Violation (< {min_green_s:.0f}s)"
        if not bool(row["ended_by_switch"]):
            return "Active / Episode End"
        return f"Compliant (≥ {min_green_s:.0f}s)"

    plot_df["status"] = plot_df.apply(_compliance_status, axis=1)

    bars = (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X("block_id:O", title="Green Phase Sequence #"),
            y=alt.Y("duration_s:Q", title="Green Duration (s)"),
            color=alt.Color(
                "status:N",
                title="Shield Constraint",
                scale=alt.Scale(
                    domain=[
                        f"Compliant (≥ {min_green_s:.0f}s)",
                        "Active / Episode End",
                        f"Violation (< {min_green_s:.0f}s)",
                    ],
                    range=["#10b981", "#64748b", "#ef4444"],
                ),
            ),
            tooltip=[
                alt.Tooltip("block_id:O", title="Block #"),
                alt.Tooltip("phase_label:N", title="Phase"),
                alt.Tooltip("start_s:Q", title="Start (s)", format=".1f"),
                alt.Tooltip("end_s:Q", title="End (s)", format=".1f"),
                alt.Tooltip("duration_s:Q", title="Duration (s)", format=".1f"),
                alt.Tooltip("status:N", title="Constraint Status"),
            ],
        )
    )

    rule_df = pd.DataFrame([{"min_green": float(min_green_s), "label": f"t_min_green = {min_green_s:.0f}s"}])
    rule = (
        alt.Chart(rule_df)
        .mark_rule(color="#ef4444", strokeDash=[6, 4], strokeWidth=2.0)
        .encode(y="min_green:Q")
    )

    return alt.layer(bars, rule).properties(height=height)


def build_proposed_vs_executed_chart(audit_df: pd.DataFrame, height: int = 250) -> alt.Chart:
    """Plot proposed vs executed phase actions and highlight safety shield interventions."""
    long_df = pd.concat(
        [
            audit_df[["sim_time", "proposed_action"]].rename(columns={"proposed_action": "action"}).assign(
                action_type="Proposed Action"
            ),
            audit_df[["sim_time", "executed_action"]].rename(columns={"executed_action": "action"}).assign(
                action_type="Executed Action"
            ),
        ],
        ignore_index=True,
    )

    lines = (
        alt.Chart(long_df)
        .mark_line(interpolate="step-after", strokeWidth=2.0)
        .encode(
            x=alt.X("sim_time:Q", title="Simulation Time (s)"),
            y=alt.Y("action:Q", title="Phase Index", axis=alt.Axis(tickMinStep=1)),
            color=alt.Color(
                "action_type:N",
                title="Action Stream",
                scale=alt.Scale(
                    domain=["Proposed Action", "Executed Action"],
                    range=["#f59e0b", "#2563eb"],
                ),
            ),
            strokeDash=alt.StrokeDash(
                "action_type:N",
                legend=None,
                scale=alt.Scale(
                    domain=["Proposed Action", "Executed Action"],
                    range=[[4, 3], [1, 0]],
                ),
            ),
            tooltip=[
                alt.Tooltip("sim_time:Q", title="Sim Time (s)", format=".1f"),
                alt.Tooltip("action_type:N", title="Stream"),
                alt.Tooltip("action:Q", title="Phase"),
            ],
        )
    )

    blocked = audit_df[audit_df["shield_intervened"]].copy()
    if blocked.empty:
        return lines.properties(height=height).interactive()

    markers = (
        alt.Chart(blocked)
        .mark_point(shape="cross", size=110, strokeWidth=2.5, color="#ef4444")
        .encode(
            x="sim_time:Q",
            y="proposed_action:Q",
            tooltip=[
                alt.Tooltip("sim_time:Q", title="Blocked at (s)", format=".1f"),
                alt.Tooltip("active_phase:Q", title="Active Phase"),
                alt.Tooltip("elapsed_green_s:Q", title="Elapsed Green (s)", format=".1f"),
                alt.Tooltip("proposed_action:Q", title="Proposed (Blocked)"),
                alt.Tooltip("executed_action:Q", title="Executed (Shield Hold)"),
            ],
        )
    )
    return alt.layer(lines, markers).properties(height=height)
