from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.contracts import TELEMETRY_COLUMNS, load_default_config
from src.utils.controllers import (
    CONTROLLER_SPECS,
    ControllerAdapter,
    ControllerAvailability,
    build_controller,
    discover_checkpoints,
    discover_networks_and_routes,
    get_controller_availabilities,
    infer_network_and_demand,
    make_traffic_env,
    resolve_controller_key,
)
from src.utils.telemetry import (
    TelemetryRecorder,
    compute_episode_kpis,
    validate_telemetry_columns,
)


KPI_METADATA: tuple[tuple[str, str, str, str], ...] = (
    ("avg_vehicle_delay_s_per_veh", "Avg Vehicle Delay", "s/veh", "inverse"),
    ("max_lane_wait_s", "Max Lane Wait", "s", "inverse"),
    ("lane_wait_variance_s2", "Lane Wait Variance", "s²", "inverse"),
    ("max_queue_vehicles", "Max Intersection Queue", "vehicles", "inverse"),
    ("throughput_veh_per_hr", "Intersection Throughput", "veh/hr", "normal"),
    ("total_co2_g_per_km", "Total CO₂ Emissions", "g/km", "inverse"),
    ("oscillation_switches_per_hr", "Action Oscillation Freq", "switches/hr", "inverse"),
)


@dataclass(frozen=True)
class RecordedLogEntry:
    """Discovered recorded telemetry run inside outputs/."""

    run_name: str
    telemetry_path: Path
    lanes_path: Path | None
    summary_json_path: Path | None
    summary_csv_path: Path | None
    model_name: str
    network: str
    demand_regime: str
    seed: int | None
    display_label: str


@dataclass
class LoadedEpisodeView:
    """Unified episode telemetry payload produced by Recorded Log and Live modes."""

    run_name: str
    source_mode: str
    model_name: str
    network: str
    demand_regime: str
    seed: int
    telemetry_df: pd.DataFrame
    lane_df: pd.DataFrame | None
    step_distance_m: list[float] | None
    summary: dict[str, Any] | None
    metadata: dict[str, Any] = field(default_factory=dict)
    source_path: Path | None = None


def _derive_run_name(path: Path) -> str:
    stem = path.stem
    for suffix in ("_telemetry", "-telemetry"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _is_companion_file(path: Path) -> bool:
    stem = path.stem.lower()
    return stem.endswith("_lanes") or stem.endswith("_summary") or stem.endswith("-lanes") or stem.endswith("-summary")


def _read_schema_columns(path: Path) -> list[str]:
    if path.suffix.lower() == ".parquet":
        return list(pq.read_schema(path).names)
    return list(pd.read_csv(path, nrows=0).columns)


def _find_companion_file(base_dir: Path, run_name: str, suffix: str, extensions: tuple[str, ...]) -> Path | None:
    for ext in extensions:
        candidate = base_dir / f"{run_name}_{suffix}{ext}"
        if candidate.exists():
            return candidate
    return None


def _read_summary_metadata(
    summary_json_path: Path | None,
    summary_csv_path: Path | None,
    fallback_name: str,
) -> tuple[str, str, str, int | None]:
    if summary_json_path and summary_json_path.exists():
        try:
            with open(summary_json_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            summary = payload.get("summary", payload)
            return (
                str(summary.get("model_name", fallback_name)),
                str(summary.get("network", "unknown")),
                str(summary.get("demand_regime", "unknown")),
                int(summary["seed"]) if summary.get("seed") is not None else None,
            )
        except Exception:
            pass

    if summary_csv_path and summary_csv_path.exists():
        try:
            sdf = pd.read_csv(summary_csv_path, nrows=1)
            if not sdf.empty:
                row = sdf.iloc[0]
                return (
                    str(row.get("model_name", fallback_name)),
                    str(row.get("network", "unknown")),
                    str(row.get("demand_regime", "unknown")),
                    int(row["seed"]) if pd.notna(row.get("seed")) else None,
                )
        except Exception:
            pass

    return fallback_name, "unknown", "unknown", None


def discover_recorded_logs(
    outputs_dir: str | Path = "outputs",
) -> tuple[list[RecordedLogEntry], list[tuple[Path, str]]]:
    """Discover and schema-validate exported telemetry files in outputs/."""
    root = Path(outputs_dir)
    if not root.exists():
        return [], []

    candidates: list[Path] = []
    for ext in ("*.csv", "*.parquet"):
        for path in sorted(root.rglob(ext)):
            if not _is_companion_file(path):
                candidates.append(path)

    # Deduplicate when both CSV and Parquet exist for the same run_name (prefer Parquet or CSV deterministically).
    valid_entries: list[RecordedLogEntry] = []
    invalid_files: list[tuple[Path, str]] = []
    seen_runs: dict[tuple[Path, str], RecordedLogEntry] = {}

    for path in candidates:
        try:
            cols = _read_schema_columns(path)
            missing = [c for c in TELEMETRY_COLUMNS if c not in cols]
            if missing:
                invalid_files.append((path, f"Missing TELEMETRY_COLUMNS: {missing}"))
                continue
        except Exception as exc:
            invalid_files.append((path, f"Failed to read schema: {exc}"))
            continue

        run_name = _derive_run_name(path)
        run_key = (path.parent.resolve(), run_name)
        if run_key in seen_runs and path.suffix.lower() != ".parquet":
            continue

        lanes_path = _find_companion_file(path.parent, run_name, "lanes", (".parquet", ".csv"))
        summary_json_path = _find_companion_file(path.parent, run_name, "summary", (".json",))
        summary_csv_path = _find_companion_file(path.parent, run_name, "summary", (".csv",))

        model_name, network, demand, seed = _read_summary_metadata(
            summary_json_path, summary_csv_path, run_name
        )
        seed_str = f"seed={seed}" if seed is not None else "seed=?"
        display_label = f"{model_name} | {network}/{demand} | {seed_str} [{path.name}]"

        entry = RecordedLogEntry(
            run_name=run_name,
            telemetry_path=path,
            lanes_path=lanes_path,
            summary_json_path=summary_json_path,
            summary_csv_path=summary_csv_path,
            model_name=model_name,
            network=network,
            demand_regime=demand,
            seed=seed,
            display_label=display_label,
        )
        seen_runs[run_key] = entry

    valid_entries = sorted(seen_runs.values(), key=lambda e: (e.model_name, e.run_name))
    return valid_entries, invalid_files


def _coerce_telemetry_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    int_cols = ("episode", "active_phase", "proposed_action", "executed_action", "step_throughput")
    float_cols = (
        "sim_time",
        "elapsed_green_s",
        "total_queue",
        "max_lane_wait_s",
        "lane_wait_variance_s2",
        "step_co2_mg",
        "reward",
    )
    bool_cols = ("mask_approved", "phase_switched")

    for col in int_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)
    for col in float_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0).astype(float)
    for col in bool_cols:
        if out[col].dtype == object:
            out[col] = out[col].astype(str).str.lower().isin({"true", "1", "yes"})
        else:
            out[col] = out[col].astype(bool)
    out["ts_id"] = out["ts_id"].astype(str)
    return out.sort_values("sim_time").reset_index(drop=True)


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_recorded_log(entry: RecordedLogEntry, cfg: dict[str, Any] | None = None) -> LoadedEpisodeView:
    """Load and validate a recorded telemetry episode and its companion lane/summary files."""
    cfg = cfg or load_default_config()
    raw_df = _read_table(entry.telemetry_path)
    validate_telemetry_columns(raw_df)
    telemetry_df = _coerce_telemetry_types(raw_df[list(TELEMETRY_COLUMNS)])

    lane_df: pd.DataFrame | None = None
    if entry.lanes_path and entry.lanes_path.exists():
        lane_df = _read_table(entry.lanes_path)
        if "sim_time" in lane_df.columns:
            lane_df["sim_time"] = pd.to_numeric(lane_df["sim_time"], errors="coerce")

    summary: dict[str, Any] | None = None
    metadata: dict[str, Any] = {
        "delta_time_s": float(cfg["shield"]["delta_time_s"]),
        "min_green_s": float(cfg["shield"]["min_green_s"]),
        "yellow_s": float(cfg["shield"]["yellow_s"]),
        "all_red_s": float(cfg["shield"]["all_red_s"]),
        "w_thresh_s": float(cfg["reward"]["w_thresh_s"]),
    }
    step_distance_m: list[float] | None = None

    if entry.summary_json_path and entry.summary_json_path.exists():
        with open(entry.summary_json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        summary = payload.get("summary", payload)
        metadata.update(payload.get("metadata", {}))
        distances = payload.get("step_distance_m")
        if isinstance(distances, list) and len(distances) == len(telemetry_df):
            step_distance_m = [float(x) for x in distances]
    elif entry.summary_csv_path and entry.summary_csv_path.exists():
        sdf = pd.read_csv(entry.summary_csv_path, nrows=1)
        if not sdf.empty:
            summary = sdf.iloc[0].to_dict()

    if step_distance_m is None and "step_distance_m" in raw_df.columns:
        step_distance_m = pd.to_numeric(raw_df["step_distance_m"], errors="coerce").fillna(0.0).tolist()

    return LoadedEpisodeView(
        run_name=entry.run_name,
        source_mode="recorded",
        model_name=entry.model_name,
        network=entry.network,
        demand_regime=entry.demand_regime,
        seed=entry.seed if entry.seed is not None else 0,
        telemetry_df=telemetry_df,
        lane_df=lane_df,
        step_distance_m=step_distance_m,
        summary=summary,
        metadata=metadata,
        source_path=entry.telemetry_path,
    )


def slice_episode_view(
    view: LoadedEpisodeView,
    max_sim_time: float | None = None,
    min_sim_time: float | None = None,
) -> LoadedEpisodeView:
    """Slice an episode view to a simulation-time window for replay scrubbing."""
    df = view.telemetry_df
    if df.empty or (max_sim_time is None and min_sim_time is None):
        return view

    mask = pd.Series(True, index=df.index)
    if min_sim_time is not None:
        mask &= df["sim_time"] >= float(min_sim_time)
    if max_sim_time is not None:
        mask &= df["sim_time"] <= float(max_sim_time)

    sliced_df = df.loc[mask].reset_index(drop=True)
    sliced_distances: list[float] | None = None
    if view.step_distance_m is not None and len(view.step_distance_m) == len(df):
        sliced_distances = [d for d, keep in zip(view.step_distance_m, mask.tolist()) if keep]

    sliced_lanes: pd.DataFrame | None = None
    if view.lane_df is not None and not view.lane_df.empty and "sim_time" in view.lane_df.columns:
        l_mask = pd.Series(True, index=view.lane_df.index)
        if min_sim_time is not None:
            l_mask &= view.lane_df["sim_time"] >= float(min_sim_time)
        if max_sim_time is not None:
            l_mask &= view.lane_df["sim_time"] <= float(max_sim_time)
        sliced_lanes = view.lane_df.loc[l_mask].reset_index(drop=True)

    return LoadedEpisodeView(
        run_name=view.run_name,
        source_mode=view.source_mode,
        model_name=view.model_name,
        network=view.network,
        demand_regime=view.demand_regime,
        seed=view.seed,
        telemetry_df=sliced_df,
        lane_df=sliced_lanes,
        step_distance_m=sliced_distances,
        summary=view.summary if len(sliced_df) == len(df) else None,
        metadata=view.metadata,
        source_path=view.source_path,
    )


def compute_view_kpis(view: LoadedEpisodeView) -> dict[str, float | None]:
    """Compute the 7 evaluation KPIs for a full or scrubbed episode view."""
    delta_time_s = float(view.metadata.get("delta_time_s", 5.0))
    return compute_episode_kpis(
        view.telemetry_df,
        lane_df=view.lane_df,
        step_distance_m=view.step_distance_m,
        fallback_summary=view.summary,
        delta_time_s=delta_time_s,
    )


def compute_kpi_deltas(
    current_kpis: dict[str, float | None],
    baseline_kpis: dict[str, float | None] | None,
) -> dict[str, dict[str, float | None]]:
    """Compute raw and percentage deltas vs a baseline KPI dictionary."""
    deltas: dict[str, dict[str, float | None]] = {}
    for key, _, _, _ in KPI_METADATA:
        cur = current_kpis.get(key)
        base = baseline_kpis.get(key) if baseline_kpis else None
        if cur is None or base is None:
            deltas[key] = {"diff": None, "pct": None}
            continue
        diff = float(cur) - float(base)
        pct = (100.0 * diff / float(base)) if float(base) != 0.0 else None
        deltas[key] = {"diff": diff, "pct": pct}
    return deltas


class LiveRolloutSession:
    """Incremental headless SUMO session feeding live step telemetry to the dashboard."""

    def __init__(self) -> None:
        self.env: Any = None
        self.controller: ControllerAdapter | None = None
        self.recorder = TelemetryRecorder(episode=1, delta_time_s=5.0)
        self.cfg: dict[str, Any] = load_default_config()
        self.obs: Any = None
        self.info: dict[str, Any] | None = None
        self.done: bool = False
        self.is_running: bool = False
        self.is_paused: bool = False
        self.error_message: str | None = None

        self.controller_key: str = "fixed_time"
        self.model_name: str = CONTROLLER_SPECS["fixed_time"].display_name
        self.checkpoint_path: str | None = None
        self.net_file: str = ""
        self.route_file: str = ""
        self.network_name: str = "single_intersection"
        self.demand_regime: str = "vhvh"
        self.seed: int = 42
        self.num_seconds: int = 300
        self.delta_time_s: int = 5
        self.use_shield: bool = True
        self.reward_mode: str = "starvation"
        self.run_name: str = ""
        self.exported_files: dict[str, Path] | None = None

    def close_env(self) -> None:
        """Safely close any active SUMO environment connection."""
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
            finally:
                self.env = None

    def start(
        self,
        controller_key: str,
        net_file: str | Path,
        route_file: str | Path,
        seed: int = 42,
        num_seconds: int = 300,
        checkpoint_path: str | Path | None = None,
        use_shield: bool | None = None,
        reward_mode: str | None = None,
        allow_untrained: bool = False,
        config_path: str | Path = "configs/default.json",
    ) -> None:
        """Initialize a fresh SafetyShieldedTrafficEnv and controller for live stepping."""
        self.close_env()
        self.error_message = None
        self.exported_files = None

        self.cfg = load_default_config(str(config_path))
        resolved_key = resolve_controller_key(controller_key)
        spec = CONTROLLER_SPECS[resolved_key]

        self.controller_key = resolved_key
        self.model_name = spec.display_name
        self.checkpoint_path = str(checkpoint_path) if checkpoint_path else None
        self.net_file = str(net_file)
        self.route_file = str(route_file)
        self.network_name, self.demand_regime = infer_network_and_demand(net_file, route_file)
        self.seed = int(seed)
        self.num_seconds = int(num_seconds)
        self.delta_time_s = int(self.cfg["shield"]["delta_time_s"])
        self.use_shield = spec.default_use_shield if use_shield is None else bool(use_shield)
        self.reward_mode = spec.default_reward_mode if reward_mode is None else str(reward_mode)
        self.run_name = (
            f"live_{self.controller_key}_{self.network_name}_{self.demand_regime}_seed{self.seed}"
        )

        self.recorder = TelemetryRecorder(episode=1, delta_time_s=float(self.delta_time_s))
        try:
            self.env = make_traffic_env(
                cfg=self.cfg,
                net_file=self.net_file,
                route_file=self.route_file,
                use_shield=self.use_shield,
                reward_mode=self.reward_mode,
                num_seconds=self.num_seconds,
                delta_time_s=self.delta_time_s,
                validate_contracts=True,
            )
            self.obs, self.info = self.env.reset(seed=self.seed)
            initial_info = self.info[self.env.ts_id] if self.env.is_multi_agent else self.info
            self.recorder.reset(episode=1, initial_info=initial_info)
            self.controller = build_controller(
                controller_key=self.controller_key,
                num_actions=self.env.num_actions_map[self.env.ts_id],
                cfg=self.cfg,
                checkpoint_path=self.checkpoint_path,
                allow_untrained=allow_untrained,
            )
            self.done = False
            self.is_running = True
            self.is_paused = False
        except Exception as exc:
            self.close_env()
            self.is_running = False
            self.is_paused = False
            self.done = False
            self.error_message = str(exc)
            raise

    def step_n(self, num_steps: int = 1) -> int:
        """Advance up to num_steps decision intervals in the live SUMO environment."""
        if self.env is None or self.controller is None or self.done:
            self.is_running = False
            return 0

        steps_taken = 0
        try:
            for _ in range(max(1, int(num_steps))):
                if self.done or self.env is None:
                    break
                if self.env.is_multi_agent:
                    action_map = {
                        tid: self.controller.select_action(self.obs[tid], self.info[tid], self.env)
                        for tid in self.env.ts_ids
                    }
                    obs, reward, terminated, truncated, info = self.env.step(action_map)
                    focal_tid = self.env.ts_id
                    self.recorder.record_step(info=info[focal_tid], reward=float(reward[focal_tid]), episode=1)
                    self.done = bool(terminated["__all__"] or truncated["__all__"])
                else:
                    action = self.controller.select_action(self.obs, self.info, self.env)
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    self.recorder.record_step(info=info, reward=reward, episode=1)
                    self.done = bool(terminated or truncated)
                self.obs = obs
                self.info = info
                steps_taken += 1
                if self.done:
                    self.is_running = False
                    self.is_paused = False
                    self.close_env()
                    break
        except Exception as exc:
            self.close_env()
            self.is_running = False
            self.is_paused = False
            self.error_message = str(exc)
            raise
        return steps_taken

    def pause(self) -> None:
        if self.is_running and not self.done:
            self.is_running = False
            self.is_paused = True

    def resume(self) -> None:
        if self.env is not None and not self.done:
            self.is_running = True
            self.is_paused = False

    def stop(self) -> None:
        self.close_env()
        self.is_running = False
        self.is_paused = False
        self.done = True

    def export_to_outputs(
        self,
        output_dir: str | Path = "outputs",
        file_format: str = "csv",
    ) -> dict[str, Path]:
        """Export recorded live rollout telemetry and summary into outputs/."""
        if len(self.recorder) == 0:
            raise ValueError("No telemetry steps recorded yet to export.")
        metadata = {
            "controller_key": self.controller_key,
            "model_name": self.model_name,
            "checkpoint_path": self.checkpoint_path,
            "net_file": self.net_file,
            "route_file": self.route_file,
            "use_shield": self.use_shield,
            "reward_mode": self.reward_mode,
            "seed": self.seed,
            "num_seconds": self.num_seconds,
            "min_green_s": float(self.cfg["shield"]["min_green_s"]),
            "yellow_s": float(self.cfg["shield"]["yellow_s"]),
            "all_red_s": float(self.cfg["shield"]["all_red_s"]),
            "w_thresh_s": float(self.cfg["reward"]["w_thresh_s"]),
        }
        self.exported_files = self.recorder.export(
            output_dir=output_dir,
            run_name=self.run_name,
            model_name=self.model_name,
            network=self.network_name,
            demand_regime=self.demand_regime,
            seed=self.seed,
            file_format=file_format,
            metadata=metadata,
        )
        return self.exported_files

    def to_episode_view(self) -> LoadedEpisodeView:
        """Convert the live TelemetryRecorder buffer into a LoadedEpisodeView."""
        t_df = self.recorder.telemetry_dataframe()
        l_df = self.recorder.lane_dataframe()
        summary = (
            self.recorder.compute_episode_summary(
                model_name=self.model_name,
                network=self.network_name,
                demand_regime=self.demand_regime,
                seed=self.seed,
            )
            if len(self.recorder) > 0
            else None
        )
        metadata = {
            "controller_key": self.controller_key,
            "checkpoint_path": self.checkpoint_path,
            "use_shield": self.use_shield,
            "reward_mode": self.reward_mode,
            "delta_time_s": float(self.delta_time_s),
            "min_green_s": float(self.cfg["shield"]["min_green_s"]),
            "yellow_s": float(self.cfg["shield"]["yellow_s"]),
            "all_red_s": float(self.cfg["shield"]["all_red_s"]),
            "w_thresh_s": float(self.cfg["reward"]["w_thresh_s"]),
        }
        return LoadedEpisodeView(
            run_name=self.run_name or self.model_name,
            source_mode="live",
            model_name=self.model_name,
            network=self.network_name,
            demand_regime=self.demand_regime,
            seed=self.seed,
            telemetry_df=t_df,
            lane_df=l_df,
            step_distance_m=self.recorder.step_distances_m(),
            summary=summary,
            metadata=metadata,
            source_path=None,
        )


def get_dashboard_environment_status(
    outputs_dir: str | Path = "outputs",
    networks_dir: str | Path = "networks",
) -> dict[str, Any]:
    """Gather available recorded logs, controllers, checkpoints, and SUMO network/route files."""
    availabilities: dict[str, ControllerAvailability] = get_controller_availabilities()
    checkpoints = discover_checkpoints()
    net_files, route_files = discover_networks_and_routes(networks_dir)
    recorded_logs, invalid_logs = discover_recorded_logs(outputs_dir)
    return {
        "availabilities": availabilities,
        "checkpoints": checkpoints,
        "net_files": net_files,
        "route_files": route_files,
        "recorded_logs": recorded_logs,
        "invalid_logs": invalid_logs,
    }
