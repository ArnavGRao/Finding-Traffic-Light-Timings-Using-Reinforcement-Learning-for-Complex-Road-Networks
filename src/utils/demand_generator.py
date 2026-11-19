"""Procedural continuous-spectrum SUMO demand generator for single_intersection and resco_4x4."""

from __future__ import annotations

import functools
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
RESCO_UPSTREAM_ROUTE_FILE = REPO_ROOT / "networks/resco_4x4/grid4x4_1.rou.xml"

ARCHETYPES = ("balanced", "asymmetric", "tidal", "hybrid_intermediate")

SINGLE_INTERSECTION_ROUTES: tuple[tuple[str, str], ...] = (
    ("route_ns", "n_t t_s"),
    ("route_nw", "n_t t_w"),
    ("route_ne", "n_t t_e"),
    ("route_we", "w_t t_e"),
    ("route_wn", "w_t t_n"),
    ("route_ws", "w_t t_s"),
    ("route_ew", "e_t t_w"),
    ("route_en", "e_t t_n"),
    ("route_es", "e_t t_s"),
    ("route_sn", "s_t t_n"),
    ("route_se", "s_t t_e"),
    ("route_sw", "s_t t_w"),
)

# (approach, movement, route_id, axis, movement_share)
SINGLE_INTERSECTION_MOVEMENTS: tuple[tuple[str, str, str, str, float], ...] = (
    ("n", "straight", "route_ns", "ns", 2.0 / 3.0),
    ("n", "right", "route_nw", "ns", 1.0 / 6.0),
    ("n", "left", "route_ne", "ns", 1.0 / 6.0),
    ("s", "straight", "route_sn", "ns", 2.0 / 3.0),
    ("s", "right", "route_se", "ns", 1.0 / 6.0),
    ("s", "left", "route_sw", "ns", 1.0 / 6.0),
    ("w", "straight", "route_we", "ew", 2.0 / 3.0),
    ("w", "right", "route_wn", "ew", 1.0 / 6.0),
    ("w", "left", "route_ws", "ew", 1.0 / 6.0),
    ("e", "straight", "route_ew", "ew", 2.0 / 3.0),
    ("e", "right", "route_es", "ew", 1.0 / 6.0),
    ("e", "left", "route_en", "ew", 1.0 / 6.0),
)

SINGLE_BASELINE_TOTAL_VPH = 2400.0  # 4 approaches * 600 veh/hr
RESCO_BASELINE_TOTAL_VPH = 1500.0   # ~1473 veh/hr in upstream grid4x4_1.rou.xml


@dataclass(frozen=True)
class CorridorRoute:
    edges: str
    entry_edge: str
    exit_edge: str
    axis: str  # "ns" or "ew"


@dataclass(frozen=True)
class DemandSegment:
    begin: float
    end: float
    volume_mult: float
    ns_ratio: float
    surge_targets: tuple[str, ...] = ()
    surge_boost: float = 1.0


def _edge_axis_from_name(edge_id: str) -> str | None:
    lower = edge_id.lower()
    if "top" in lower or "bottom" in lower:
        return "ns"
    if "left" in lower or "right" in lower:
        return "ew"
    # Interior RESCO edges look like A0A1 (NS: same col) or A0B0 (EW: same row).
    if len(edge_id) == 4 and edge_id[0].isalpha() and edge_id[2].isalpha():
        if edge_id[0] == edge_id[2]:
            return "ns"
        if edge_id[1] == edge_id[3]:
            return "ew"
    return None


def _classify_resco_route(edges_str: str) -> CorridorRoute:
    edge_list = edges_str.split()
    entry_edge = edge_list[0]
    exit_edge = edge_list[-1]
    entry_axis = _edge_axis_from_name(entry_edge) or "ns"
    exit_axis = _edge_axis_from_name(exit_edge) or entry_axis

    if entry_axis == exit_axis:
        axis = entry_axis
    else:
        ns_steps = sum(1 for e in edge_list if _edge_axis_from_name(e) == "ns")
        ew_steps = sum(1 for e in edge_list if _edge_axis_from_name(e) == "ew")
        if ns_steps > ew_steps:
            axis = "ns"
        elif ew_steps > ns_steps:
            axis = "ew"
        else:
            axis = entry_axis

    return CorridorRoute(
        edges=edges_str,
        entry_edge=entry_edge,
        exit_edge=exit_edge,
        axis=axis,
    )


@functools.lru_cache(maxsize=1)
def load_resco_route_pool() -> tuple[tuple[CorridorRoute, ...], tuple[CorridorRoute, ...]]:
    """Parse and classify all valid corridor routes from networks/resco_4x4/grid4x4_1.rou.xml."""
    if not RESCO_UPSTREAM_ROUTE_FILE.exists():
        raise FileNotFoundError(f"RESCO upstream route file not found: {RESCO_UPSTREAM_ROUTE_FILE}")

    tree = ET.parse(RESCO_UPSTREAM_ROUTE_FILE)
    root = tree.getroot()
    unique_edges: list[str] = []
    seen: set[str] = set()
    for veh in root.findall("vehicle"):
        route_el = veh.find("route")
        if route_el is None:
            continue
        edges_str = route_el.attrib.get("edges", "").strip()
        if edges_str and edges_str not in seen:
            seen.add(edges_str)
            unique_edges.append(edges_str)

    ns_routes: list[CorridorRoute] = []
    ew_routes: list[CorridorRoute] = []
    for edges_str in unique_edges:
        cr = _classify_resco_route(edges_str)
        if cr.axis == "ns":
            ns_routes.append(cr)
        else:
            ew_routes.append(cr)

    if not ns_routes or not ew_routes:
        raise RuntimeError(
            f"Failed to classify RESCO routes into NS/EW pools (ns={len(ns_routes)}, ew={len(ew_routes)})."
        )
    return tuple(ns_routes), tuple(ew_routes)


def _sample_breakpoints(rng: np.random.Generator, num_seconds: int, num_segments: int) -> list[float]:
    if num_segments <= 1:
        return [0.0, float(num_seconds)]
    min_seg = max(10.0, float(num_seconds) / (num_segments * 2.5))
    if min_seg * num_segments >= float(num_seconds):
        return [round(float(num_seconds) * i / num_segments, 2) for i in range(num_segments + 1)]

    slack = float(num_seconds) - min_seg * num_segments
    cuts = np.sort(rng.uniform(0.0, slack, size=num_segments - 1))
    points = [0.0]
    for idx, c in enumerate(cuts, start=1):
        points.append(round(float(idx * min_seg + c), 2))
    points.append(float(num_seconds))
    return points


def _build_segments(
    rng: np.random.Generator,
    archetype: str,
    num_seconds: int,
    available_targets: tuple[str, ...],
    *,
    ns_ratio_override: float | None = None,
    explicit_segments: list[DemandSegment] | None = None,
) -> list[DemandSegment]:
    if explicit_segments is not None:
        return explicit_segments

    if archetype == "balanced":
        ratio = float(ns_ratio_override) if ns_ratio_override is not None else float(rng.uniform(0.45, 0.55))
        return [DemandSegment(begin=0.0, end=float(num_seconds), volume_mult=1.0, ns_ratio=ratio)]

    if archetype == "asymmetric":
        if ns_ratio_override is not None:
            ratio = float(ns_ratio_override)
        else:
            dom = float(rng.uniform(0.60, 0.85))
            ratio = dom if bool(rng.integers(0, 2)) else (1.0 - dom)
        return [DemandSegment(begin=0.0, end=float(num_seconds), volume_mult=1.0, ns_ratio=ratio)]

    if archetype == "tidal":
        num_segments = int(rng.integers(2, 5))  # 2 to 4 intervals
        points = _sample_breakpoints(rng, num_seconds, num_segments)
        modes = ["ns", "balanced", "ew"]
        prev_mode: str | None = None
        segments: list[DemandSegment] = []
        for idx in range(num_segments):
            candidates = [m for m in modes if m != prev_mode]
            mode = str(candidates[int(rng.integers(0, len(candidates)))])
            prev_mode = mode
            if mode == "ns":
                ratio = float(rng.uniform(0.65, 0.85))
            elif mode == "ew":
                ratio = float(rng.uniform(0.15, 0.35))
            else:
                ratio = float(rng.uniform(0.45, 0.55))
            vol_mult = float(rng.uniform(0.85, 1.15))
            segments.append(
                DemandSegment(
                    begin=points[idx],
                    end=points[idx + 1],
                    volume_mult=vol_mult,
                    ns_ratio=ratio,
                )
            )
        return segments

    if archetype == "hybrid_intermediate":
        num_stages = int(rng.integers(3, 6))  # 3 to 5 gradual ramp stages
        stage_points = _sample_breakpoints(rng, num_seconds, num_stages)
        start_ratio = float(rng.uniform(0.20, 0.80))
        end_ratio = float(rng.uniform(0.20, 0.80))
        start_vol = float(rng.uniform(0.70, 1.05))
        peak_vol = float(rng.uniform(1.05, 1.30))

        # Schedule 1-2 localized short-duration surges on random approaches/corridors
        num_surges = int(rng.integers(1, 3))
        surge_windows: list[tuple[float, float, tuple[str, ...], float]] = []
        for _ in range(num_surges):
            dur = float(rng.uniform(0.15, 0.30) * num_seconds)
            s_begin = round(float(rng.uniform(0.0, max(1.0, num_seconds - dur))), 2)
            s_end = round(min(float(num_seconds), s_begin + dur), 2)
            if s_end <= s_begin + 1.0:
                continue
            k_targets = min(len(available_targets), int(rng.integers(1, min(4, len(available_targets)) + 1)))
            chosen = tuple(str(x) for x in rng.choice(available_targets, size=k_targets, replace=False))
            boost = float(rng.uniform(1.35, 1.85))
            surge_windows.append((s_begin, s_end, chosen, boost))

        # Partition timeline at all stage and surge boundaries so emitted flows are strictly non-overlapping
        all_pts = sorted({*stage_points, *[w[0] for w in surge_windows], *[w[1] for w in surge_windows]})
        merged_pts: list[float] = [all_pts[0]]
        for p in all_pts[1:]:
            if p - merged_pts[-1] >= 1.0:
                merged_pts.append(p)
        if merged_pts[-1] < float(num_seconds):
            merged_pts[-1] = float(num_seconds)

        segments = []
        for idx in range(len(merged_pts) - 1):
            b, e = merged_pts[idx], merged_pts[idx + 1]
            mid_frac = ((b + e) * 0.5) / max(1.0, float(num_seconds))
            ratio = float(np.clip(start_ratio + mid_frac * (end_ratio - start_ratio), 0.15, 0.85))
            # Smooth triangular ramp from start_vol -> peak_vol -> start_vol
            ramp = 1.0 - abs(2.0 * mid_frac - 1.0)
            vol_mult = float(start_vol + ramp * (peak_vol - start_vol))

            active_targets: list[str] = []
            max_boost = 1.0
            for sb, se, targets, boost in surge_windows:
                if (b + e) * 0.5 >= sb and (b + e) * 0.5 <= se:
                    active_targets.extend(targets)
                    max_boost = max(max_boost, boost)

            segments.append(
                DemandSegment(
                    begin=b,
                    end=e,
                    volume_mult=vol_mult,
                    ns_ratio=ratio,
                    surge_targets=tuple(sorted(set(active_targets))),
                    surge_boost=max_boost,
                )
            )
        return segments

    raise ValueError(f"Unsupported archetype '{archetype}'. Expected one of {ARCHETYPES}.")


def _write_single_intersection_rou(
    out_path: Path,
    segments: list[DemandSegment],
    volume_scale: float,
    archetype: str,
    seed: int,
) -> None:
    flows: list[tuple[float, float, str, str, float]] = []
    for seg_idx, seg in enumerate(segments, start=1):
        if seg.end <= seg.begin:
            continue
        seg_total = SINGLE_BASELINE_TOTAL_VPH * volume_scale * seg.volume_mult
        ns_approach_base = max(40.0, (seg_total * seg.ns_ratio) * 0.5)
        ew_approach_base = max(40.0, (seg_total * (1.0 - seg.ns_ratio)) * 0.5)

        approach_vph = {
            "n": ns_approach_base,
            "s": ns_approach_base,
            "e": ew_approach_base,
            "w": ew_approach_base,
        }
        for target in seg.surge_targets:
            if target in approach_vph:
                approach_vph[target] *= seg.surge_boost

        for approach, movement, route_id, _axis, share in SINGLE_INTERSECTION_MOVEMENTS:
            vph = max(5.0, round(approach_vph[approach] * share, 2))
            flow_id = f"{approach}_{movement}_{seg_idx}"
            flows.append((seg.begin, seg.end, flow_id, route_id, vph))

    # Strict ascending ordering by begin time (then end, flow_id)
    flows.sort(key=lambda item: (item[0], item[1], item[2]))

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f"<!-- Procedural single_intersection demand: archetype={archetype}, seed={seed}, volume_scale={volume_scale:.3f} -->",
        "<routes>",
    ]
    for route_id, edges in SINGLE_INTERSECTION_ROUTES:
        lines.append(f'    <route id="{route_id}" edges="{edges}"/>')
    lines.append("")
    for begin, end, flow_id, route_id, vph in flows:
        lines.append(
            f'    <flow id="{flow_id}" route="{route_id}" '
            f'begin="{begin:.2f}" end="{end:.2f}" vehsPerHour="{vph:.2f}" '
            f'departLane="best" departSpeed="max"/>'
        )
    lines.append("</routes>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _sample_routes_with_surge(
    rng: np.random.Generator,
    pool: tuple[CorridorRoute, ...],
    count: int,
    surge_targets: tuple[str, ...],
    surge_boost: float,
) -> list[CorridorRoute]:
    if count <= 0:
        return []
    if not surge_targets or surge_boost <= 1.0:
        indices = rng.integers(0, len(pool), size=count)
        return [pool[int(i)] for i in indices]

    target_set = set(surge_targets)
    weights = np.array(
        [surge_boost if cr.entry_edge in target_set else 1.0 for cr in pool],
        dtype=np.float64,
    )
    probs = weights / weights.sum()
    indices = rng.choice(len(pool), size=count, replace=True, p=probs)
    return [pool[int(i)] for i in indices]


def _write_resco_4x4_rou(
    out_path: Path,
    segments: list[DemandSegment],
    volume_scale: float,
    archetype: str,
    seed: int,
    rng: np.random.Generator,
) -> None:
    ns_pool, ew_pool = load_resco_route_pool()
    vehicles: list[tuple[float, str]] = []

    for seg in segments:
        dur = seg.end - seg.begin
        if dur <= 0.0:
            continue
        seg_total_vph = RESCO_BASELINE_TOTAL_VPH * volume_scale * seg.volume_mult
        ns_mean = max(1.0, seg_total_vph * seg.ns_ratio * (dur / 3600.0))
        ew_mean = max(1.0, seg_total_vph * (1.0 - seg.ns_ratio) * (dur / 3600.0))

        # Apply surge volume boost proportionally if surge corridors exist on that axis
        if seg.surge_targets and seg.surge_boost > 1.0:
            target_set = set(seg.surge_targets)
            ns_hits = sum(1 for cr in ns_pool if cr.entry_edge in target_set) / len(ns_pool)
            ew_hits = sum(1 for cr in ew_pool if cr.entry_edge in target_set) / len(ew_pool)
            ns_mean *= 1.0 + ns_hits * (seg.surge_boost - 1.0)
            ew_mean *= 1.0 + ew_hits * (seg.surge_boost - 1.0)

        ns_count = max(1, int(rng.poisson(ns_mean)))
        ew_count = max(1, int(rng.poisson(ew_mean)))

        upper_depart = max(seg.begin, seg.end - 0.5)
        ns_departs = rng.uniform(seg.begin, upper_depart, size=ns_count)
        ew_departs = rng.uniform(seg.begin, upper_depart, size=ew_count)

        ns_chosen = _sample_routes_with_surge(rng, ns_pool, ns_count, seg.surge_targets, seg.surge_boost)
        ew_chosen = _sample_routes_with_surge(rng, ew_pool, ew_count, seg.surge_targets, seg.surge_boost)

        for dep, cr in zip(ns_departs, ns_chosen):
            vehicles.append((round(float(dep), 2), cr.edges))
        for dep, cr in zip(ew_departs, ew_chosen):
            vehicles.append((round(float(dep), 2), cr.edges))

    # Strict ascending ordering by depart time
    vehicles.sort(key=lambda item: (item[0], item[1]))

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f"<!-- Procedural resco_4x4 demand: archetype={archetype}, seed={seed}, volume_scale={volume_scale:.3f}, vehicles={len(vehicles)} -->",
        '<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">',
    ]
    for veh_idx, (depart, edges) in enumerate(vehicles):
        lines.append(f'    <vehicle id="v_{veh_idx:05d}" depart="{depart:.2f}" departLane="best" departSpeed="max">')
        lines.append(f'        <route edges="{edges}"/>')
        lines.append("    </vehicle>")
    lines.append("</routes>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_random_demand_route(
    network: str,
    output_path: str | Path,
    seed: int,
    num_seconds: int = 3600,
    archetype: str | None = None,
    *,
    volume_scale: float | None = None,
    ns_ratio: float | None = None,
    explicit_segments: list[DemandSegment] | None = None,
) -> dict[str, Any]:
    """Generate a seed-reproducible SUMO .rou.xml file across the continuous demand spectrum."""
    if network not in ("single_intersection", "resco_4x4"):
        raise ValueError(f"Unsupported network '{network}'. Expected 'single_intersection' or 'resco_4x4'.")
    if num_seconds <= 0:
        raise ValueError(f"num_seconds must be positive, got {num_seconds}.")

    rng = np.random.default_rng(int(seed))
    resolved_volume_scale = (
        float(volume_scale)
        if volume_scale is not None
        else float(rng.uniform(0.50, 1.35))
    )
    resolved_archetype = (
        str(archetype)
        if archetype is not None
        else str(ARCHETYPES[int(rng.integers(0, len(ARCHETYPES)))])
    )
    if resolved_archetype not in ARCHETYPES:
        raise ValueError(f"Unsupported archetype '{resolved_archetype}'. Expected one of {ARCHETYPES}.")

    if network == "single_intersection":
        available_targets: tuple[str, ...] = ("n", "s", "e", "w")
    else:
        ns_pool, ew_pool = load_resco_route_pool()
        available_targets = tuple(sorted({cr.entry_edge for cr in (*ns_pool, *ew_pool)}))

    segments = _build_segments(
        rng=rng,
        archetype=resolved_archetype,
        num_seconds=int(num_seconds),
        available_targets=available_targets,
        ns_ratio_override=ns_ratio,
        explicit_segments=explicit_segments,
    )

    out_path = Path(output_path)
    if network == "single_intersection":
        _write_single_intersection_rou(out_path, segments, resolved_volume_scale, resolved_archetype, int(seed))
    else:
        _write_resco_4x4_rou(out_path, segments, resolved_volume_scale, resolved_archetype, int(seed), rng)

    total_dur = sum(max(0.0, s.end - s.begin) for s in segments)
    weighted_ns = (
        sum(s.ns_ratio * max(0.0, s.end - s.begin) for s in segments) / total_dur
        if total_dur > 0.0
        else 0.5
    )

    return {
        "archetype": resolved_archetype,
        "volume_scale": round(resolved_volume_scale, 4),
        "ns_ratio": round(float(weighted_ns), 4),
        "num_segments": len(segments),
        "network": network,
        "seed": int(seed),
        "num_seconds": int(num_seconds),
        "output_path": str(out_path),
    }


def generate_canonical_resco_benchmarks(
    output_dir: str | Path | None = None,
    num_seconds: int = 3600,
) -> dict[str, dict[str, Any]]:
    """Generate the three canonical 3600s held-out benchmark .rou.xml files for RESCO 4x4."""
    target_dir = Path(output_dir) if output_dir is not None else (REPO_ROOT / "networks/resco_4x4")
    results: dict[str, dict[str, Any]] = {}

    results["balanced"] = generate_random_demand_route(
        network="resco_4x4",
        output_path=target_dir / "demand_balanced_symmetric.rou.xml",
        seed=2001,
        num_seconds=num_seconds,
        archetype="balanced",
        volume_scale=1.0,
        ns_ratio=0.50,
    )
    results["asymmetric"] = generate_random_demand_route(
        network="resco_4x4",
        output_path=target_dir / "demand_asymmetric_peak_arterial.rou.xml",
        seed=2002,
        num_seconds=num_seconds,
        archetype="asymmetric",
        volume_scale=1.0,
        ns_ratio=0.78,
    )
    third = float(num_seconds) / 3.0
    tidal_segments = [
        DemandSegment(begin=0.0, end=third, volume_mult=1.0, ns_ratio=0.50),
        DemandSegment(begin=third, end=2.0 * third, volume_mult=1.0, ns_ratio=0.78),
        DemandSegment(begin=2.0 * third, end=float(num_seconds), volume_mult=1.0, ns_ratio=0.22),
    ]
    results["tidal"] = generate_random_demand_route(
        network="resco_4x4",
        output_path=target_dir / "demand_stepwise_tidal.rou.xml",
        seed=2003,
        num_seconds=num_seconds,
        archetype="tidal",
        volume_scale=1.0,
        explicit_segments=tidal_segments,
    )
    return results


if __name__ == "__main__":
    metas = generate_canonical_resco_benchmarks()
    for regime, meta in metas.items():
        print(f"Generated RESCO 4x4 {regime}: {meta}", flush=True)
