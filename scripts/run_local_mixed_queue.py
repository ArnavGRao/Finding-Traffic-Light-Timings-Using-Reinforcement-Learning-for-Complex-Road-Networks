"""Resumable sequential/parallel local queue runner for mixed-demand RL models."""

from __future__ import annotations

import argparse
import csv
import gc
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.contracts import EPISODE_SUMMARY_COLUMNS

MODEL_VARIANTS: tuple[str, ...] = (
    "vanilla_dqn",
    "d3qn_base",
    "d3qn_shield",
    "d3qn_starvation",
    "d3qn_proposed",
)

PENDING_MODELS: tuple[str, ...] = (
    "d3qn_base",
    "d3qn_shield",
    "d3qn_starvation",
    "d3qn_proposed",
)

NETWORK_CHOICES: tuple[str, ...] = ("single_intersection", "resco_4x4")
EVAL_REGIMES: tuple[str, ...] = ("balanced", "asymmetric", "tidal")

SEVEN_METRICS: tuple[str, ...] = (
    "avg_vehicle_delay_s_per_veh",
    "max_lane_wait_s",
    "lane_wait_variance_s2",
    "max_queue_vehicles",
    "throughput_veh_per_hr",
    "total_co2_g_per_km",
    "oscillation_switches_per_hr",
)

# Identical hyperparameters to Colab runner
HPARAMS: dict[str, Any] = {
    "demand": "mixed",
    "episodes": 60,
    "num_seconds": 3600,
    "seed": 42,
    "checkpoint_every": 5,
    "eval_every": 5,
    "val_seed": 999,
    "lr": 0.0005,
    "gamma": 0.99,
    "batch_size": 64,
    "buffer_capacity": 100000,
    "warmup_steps": 500,
    "eps_start": 1.0,
    "eps_end": 0.05,
    "eps_decay_steps": 10000,
    "eval_seed_start": 1000,
    "eval_num_seeds": 50,
}

STATUS_COLUMNS: tuple[str, ...] = (
    "network",
    "model",
    "status",
    "train_wall_s",
    "eval_wall_s",
    "total_wall_s",
    "best_checkpoint",
    "updated_at_utc",
    "error",
)

_PRINT_LOCK = threading.Lock()
_STATUS_LOCK = threading.Lock()


def _log(msg: str, prefix: str | None = None) -> None:
    text = f"[{prefix}] {msg}" if prefix else msg
    with _PRINT_LOCK:
        print(text, flush=True)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write_csv(path: Path, fieldnames: list[str] | tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.{threading.get_ident()}.tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in fieldnames})
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp_path, path)


def _load_existing_status(status_csv: Path) -> dict[tuple[str, str], dict[str, Any]]:
    if not status_csv.exists() or status_csv.stat().st_size == 0:
        return {}
    out: dict[tuple[str, str], dict[str, Any]] = {}
    try:
        with status_csv.open("r", newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                net = row.get("network", "").strip()
                model = row.get("model", "").strip()
                if net and model:
                    out[(net, model)] = dict(row)
    except Exception:
        return {}
    return out


def _persist_status(
    status_csv: Path,
    status_map: dict[tuple[str, str], dict[str, Any]],
    ordered_keys: list[tuple[str, str]],
) -> None:
    with _STATUS_LOCK:
        ordered_rows: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for key in ordered_keys:
            if key in status_map:
                ordered_rows.append(status_map[key])
                seen.add(key)
        for key, row in status_map.items():
            if key not in seen:
                ordered_rows.append(row)
        _atomic_write_csv(status_csv, STATUS_COLUMNS, ordered_rows)


def _expected_seed_set() -> set[int]:
    start = int(HPARAMS["eval_seed_start"])
    count = int(HPARAMS["eval_num_seeds"])
    return set(range(start, start + count))


def _has_all_eval_seeds(csv_path: Path) -> bool:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return False
    try:
        with csv_path.open("r", newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    except Exception:
        return False
    expected = _expected_seed_set()
    seen = {int(r["seed"]) for r in rows if "seed" in r and str(r["seed"]).strip() != ""}
    return expected.issubset(seen) and len(rows) >= len(expected)


def _is_training_complete(run_dir: Path, target_episodes: int) -> bool:
    best_pt = run_dir / "best.pt"
    latest_pt = run_dir / "latest.pt"
    if not (best_pt.exists() and latest_pt.exists()):
        return False
    try:
        ckpt = torch.load(str(latest_pt), map_location="cpu", weights_only=True)
        return int(ckpt.get("episode", -1)) + 1 >= int(target_episodes)
    except Exception:
        return False


def _model_paths(outputs_root: Path, network: str, model_variant: str) -> dict[str, Any]:
    ckpt_dir = outputs_root / "checkpoints" / network / "mixed" / model_variant
    run_dir = ckpt_dir / f"{model_variant}_{network}_mixed"
    train_log_dir = outputs_root / "logs" / "training" / network / "mixed" / model_variant
    eval_regime_dirs = {
        regime: outputs_root / "logs" / "eval_50_seeds" / network / regime / model_variant
        for regime in EVAL_REGIMES
    }
    return {
        "ckpt_dir": ckpt_dir,
        "run_dir": run_dir,
        "train_log_dir": train_log_dir,
        "eval_regime_dirs": eval_regime_dirs,
    }


def _is_model_complete(paths: dict[str, Any]) -> bool:
    best_pt: Path = paths["run_dir"] / "best.pt"
    if not best_pt.exists():
        return False
    eval_regime_dirs: dict[str, Path] = paths["eval_regime_dirs"]
    return all(_has_all_eval_seeds(eval_regime_dirs[r] / "combined_50_seeds.csv") for r in EVAL_REGIMES)


def _terminate_process_group(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=5)
    except Exception:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass


def _run_subprocess_with_retries(
    cmd: list[str],
    *,
    model_prefix: str,
    worker_log_path: Path,
    max_retries: int = 2,
    retry_delay_s: float = 5.0,
) -> None:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    worker_log_path.parent.mkdir(parents=True, exist_ok=True)

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 2):
        _log(f">>> (attempt {attempt}/{max_retries + 1}) {' '.join(cmd)}", prefix=model_prefix)
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        try:
            with worker_log_path.open("a", encoding="utf-8") as log_f:
                log_f.write(f"\n=== [{_utc_now()}] attempt {attempt}: {' '.join(cmd)} ===\n")
                log_f.flush()
                assert proc.stdout is not None
                for raw_line in proc.stdout:
                    line = raw_line.rstrip("\r\n")
                    if not line:
                        continue
                    log_f.write(line + "\n")
                    log_f.flush()
                    # Suppress noisy SUMO retry/step-0 lines from shared stdout; keep them in worker.log
                    if line.startswith(" Retrying in ") or line.startswith("Step #"):
                        continue
                    _log(line, prefix=model_prefix)
            ret = proc.wait()
            if ret == 0:
                return
            last_exc = subprocess.CalledProcessError(ret, cmd)
            _log(f"Subprocess exited with code {ret}.", prefix=model_prefix)
        except BaseException:
            _terminate_process_group(proc)
            raise
        finally:
            _terminate_process_group(proc)

        if attempt <= max_retries:
            _log(f"Retrying in {retry_delay_s:.0f}s with --resume...", prefix=model_prefix)
            time.sleep(retry_delay_s)

    assert last_exc is not None
    raise last_exc


def _copy_training_logs(run_dir: Path, train_log_dir: Path) -> None:
    train_log_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("training_curve.csv", "per_seed_eval.csv", "validation_curve.csv"):
        src = run_dir / filename
        if src.exists():
            shutil.copy2(src, train_log_dir / filename)


def _merge_and_summarize_regime(regime_dir: Path) -> dict[str, tuple[float, float]]:
    slice_csvs = sorted(regime_dir.glob("slice_*/*_per_seed.csv"))
    if not slice_csvs:
        raise FileNotFoundError(f"No slice per-seed CSVs found under {regime_dir}")

    rows_by_seed: dict[int, dict[str, Any]] = {}
    for csv_path in slice_csvs:
        with csv_path.open("r", newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                seed_val = int(row["seed"])
                rows_by_seed[seed_val] = row

    expected_seeds = sorted(_expected_seed_set())
    missing = sorted(_expected_seed_set() - set(rows_by_seed.keys()))
    if missing:
        raise RuntimeError(
            f"Missing evaluation seeds under {regime_dir}: {missing[:10]} (total missing={len(missing)})"
        )

    ordered_rows = [rows_by_seed[s] for s in expected_seeds]
    combined_csv = regime_dir / "combined_50_seeds.csv"
    _atomic_write_csv(combined_csv, EPISODE_SUMMARY_COLUMNS, ordered_rows)

    summary_rows: list[dict[str, Any]] = []
    metric_stats: dict[str, tuple[float, float]] = {}
    for metric in SEVEN_METRICS:
        vals = np.array([float(r[metric]) for r in ordered_rows], dtype=np.float64)
        mean_v = float(np.mean(vals))
        std_v = float(np.std(vals))
        metric_stats[metric] = (mean_v, std_v)
        summary_rows.append({"metric": metric, "mean": mean_v, "std": std_v})

    summary_csv = regime_dir / "summary_mu_sigma.csv"
    _atomic_write_csv(summary_csv, ("metric", "mean", "std"), summary_rows)
    return metric_stats


def _build_and_print_leaderboard(
    outputs_root: Path,
    network: str,
    models: list[str],
) -> Path:
    leaderboard_csv = outputs_root / "logs" / f"{network}_mixed_leaderboard.csv"
    fieldnames = ["network", "model", "regime", "n_seeds"]
    for metric in SEVEN_METRICS:
        fieldnames.extend([f"{metric}_mean", f"{metric}_std"])

    rows: list[dict[str, Any]] = []
    for model_variant in models:
        paths = _model_paths(outputs_root, network, model_variant)
        for regime in EVAL_REGIMES:
            summary_csv = paths["eval_regime_dirs"][regime] / "summary_mu_sigma.csv"
            combined_csv = paths["eval_regime_dirs"][regime] / "combined_50_seeds.csv"
            if not summary_csv.exists():
                continue
            stats: dict[str, tuple[float, float]] = {}
            with summary_csv.open("r", newline="", encoding="utf-8") as stream:
                for srow in csv.DictReader(stream):
                    stats[srow["metric"]] = (float(srow["mean"]), float(srow["std"]))
            n_seeds = 0
            if combined_csv.exists():
                with combined_csv.open("r", newline="", encoding="utf-8") as stream:
                    n_seeds = sum(1 for _ in csv.DictReader(stream))
            lrow: dict[str, Any] = {
                "network": network,
                "model": model_variant,
                "regime": regime,
                "n_seeds": n_seeds,
            }
            for metric in SEVEN_METRICS:
                mean_v, std_v = stats.get(metric, (float("nan"), float("nan")))
                lrow[f"{metric}_mean"] = round(mean_v, 4)
                lrow[f"{metric}_std"] = round(std_v, 4)
            rows.append(lrow)

    _atomic_write_csv(leaderboard_csv, fieldnames, rows)

    title = f"{network.upper().replace('_', '-')} MIXED-DEMAND 50-SEED LEADERBOARD (mean +/- std)"
    _log("\n" + "=" * 112)
    _log(title)
    _log("=" * 112)
    header = (
        f"{'Model':<16} {'Regime':<11} {'Delay(s)':>14} {'MaxWait(s)':>14} "
        f"{'WaitVar(s2)':>14} {'MaxQ':>10} {'Thru(/hr)':>14} {'CO2(g/km)':>13} {'Osc(/hr)':>11}"
    )
    _log(header)
    _log("-" * 112)
    for r in rows:
        _log(
            f"{r['model']:<16} {r['regime']:<11} "
            f"{r['avg_vehicle_delay_s_per_veh_mean']:6.2f}+/-{r['avg_vehicle_delay_s_per_veh_std']:5.2f} "
            f"{r['max_lane_wait_s_mean']:6.1f}+/-{r['max_lane_wait_s_std']:5.1f} "
            f"{r['lane_wait_variance_s2_mean']:7.1f}+/-{r['lane_wait_variance_s2_std']:5.1f} "
            f"{r['max_queue_vehicles_mean']:4.1f}+/-{r['max_queue_vehicles_std']:3.1f} "
            f"{r['throughput_veh_per_hr_mean']:6.1f}+/-{r['throughput_veh_per_hr_std']:5.1f} "
            f"{r['total_co2_g_per_km_mean']:6.1f}+/-{r['total_co2_g_per_km_std']:4.1f} "
            f"{r['oscillation_switches_per_hr_mean']:5.1f}+/-{r['oscillation_switches_per_hr_std']:4.1f}"
        )
    _log("=" * 112)
    _log(f"Saved leaderboard CSV: {leaderboard_csv}")
    return leaderboard_csv


def _run_single_model(
    *,
    outputs_root: Path,
    network: str,
    model_variant: str,
    status_csv: Path,
    status_map: dict[tuple[str, str], dict[str, Any]],
    ordered_keys: list[tuple[str, str]],
    max_retries: int,
    stagger_s: float,
) -> bool:
    """Train and 50-seed evaluate one model with per-stage resume & retry. Returns True on success."""
    paths = _model_paths(outputs_root, network, model_variant)
    ckpt_dir: Path = paths["ckpt_dir"]
    run_dir: Path = paths["run_dir"]
    train_log_dir: Path = paths["train_log_dir"]
    eval_regime_dirs: dict[str, Path] = paths["eval_regime_dirs"]
    best_ckpt = run_dir / "best.pt"
    worker_log = train_log_dir / "worker.log"
    key = (network, model_variant)

    if _is_model_complete(paths):
        _log("Already complete (best.pt + all 3x50-seed CSVs present); skipping.", prefix=model_variant)
        prev = status_map.get(key)
        if not prev or prev.get("status") != "completed":
            status_map[key] = {
                "network": network,
                "model": model_variant,
                "status": "skipped_complete",
                "train_wall_s": prev.get("train_wall_s", 0.0) if prev else 0.0,
                "eval_wall_s": prev.get("eval_wall_s", 0.0) if prev else 0.0,
                "total_wall_s": prev.get("total_wall_s", 0.0) if prev else 0.0,
                "best_checkpoint": str(best_ckpt),
                "updated_at_utc": _utc_now(),
                "error": "",
            }
            _persist_status(status_csv, status_map, ordered_keys)
        return True

    if stagger_s > 0.0:
        time.sleep(stagger_s)

    t_model_start = time.time()
    train_wall_s = 0.0
    eval_wall_s = 0.0
    status_label = "completed"
    error_msg = ""

    def _update_status(stage: str, err: str = "") -> None:
        status_map[key] = {
            "network": network,
            "model": model_variant,
            "status": stage,
            "train_wall_s": round(train_wall_s, 2),
            "eval_wall_s": round(eval_wall_s, 2),
            "total_wall_s": round(time.time() - t_model_start, 2),
            "best_checkpoint": str(best_ckpt) if best_ckpt.exists() else "",
            "updated_at_utc": _utc_now(),
            "error": err,
        }
        _persist_status(status_csv, status_map, ordered_keys)

    try:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        train_log_dir.mkdir(parents=True, exist_ok=True)

        # 1. Train (skip if all episodes already completed and best.pt exists; otherwise --resume)
        t_train = time.time()
        if _is_training_complete(run_dir, int(HPARAMS["episodes"])):
            _log("Training already at target episodes with best.pt present; resuming at evaluation.", prefix=model_variant)
        else:
            _update_status("running_train")
            train_cmd = [
                sys.executable,
                "-u",
                "-m",
                "scripts.train",
                "--model", model_variant,
                "--network", network,
                "--demand", str(HPARAMS["demand"]),
                "--episodes", str(HPARAMS["episodes"]),
                "--num-seconds", str(HPARAMS["num_seconds"]),
                "--seed", str(HPARAMS["seed"]),
                "--checkpoint-every", str(HPARAMS["checkpoint_every"]),
                "--eval-every", str(HPARAMS["eval_every"]),
                "--val-seed", str(HPARAMS["val_seed"]),
                "--lr", str(HPARAMS["lr"]),
                "--gamma", str(HPARAMS["gamma"]),
                "--batch-size", str(HPARAMS["batch_size"]),
                "--buffer-capacity", str(HPARAMS["buffer_capacity"]),
                "--warmup-steps", str(HPARAMS["warmup_steps"]),
                "--eps-start", str(HPARAMS["eps_start"]),
                "--eps-end", str(HPARAMS["eps_end"]),
                "--eps-decay-steps", str(HPARAMS["eps_decay_steps"]),
                "--output-dir", str(ckpt_dir),
                "--resume",
            ]
            _run_subprocess_with_retries(
                train_cmd,
                model_prefix=model_variant,
                worker_log_path=worker_log,
                max_retries=max_retries,
            )
        train_wall_s = time.time() - t_train

        # 2. Copy training logs into outputs/logs/training/...
        _copy_training_logs(run_dir, train_log_dir)

        if not best_ckpt.exists():
            raise FileNotFoundError(f"Expected best checkpoint not found after training: {best_ckpt}")

        # 3. 50-seed evaluation across balanced, asymmetric, tidal (per-regime skip + --resume)
        t_eval = time.time()
        seed_start = int(HPARAMS["eval_seed_start"])
        num_seeds = int(HPARAMS["eval_num_seeds"])
        slice_name = f"slice_{seed_start}_{seed_start + num_seeds - 1}"

        for regime in EVAL_REGIMES:
            regime_dir = eval_regime_dirs[regime]
            if _has_all_eval_seeds(regime_dir / "combined_50_seeds.csv"):
                _log(f"Eval regime '{regime}' already has 50 seeds; skipping.", prefix=model_variant)
                continue

            _update_status(f"running_eval:{regime}")
            slice_dir = regime_dir / slice_name
            slice_dir.mkdir(parents=True, exist_ok=True)

            eval_cmd = [
                sys.executable,
                "-u",
                "-m",
                "scripts.evaluate",
                "--model", model_variant,
                "--network", network,
                "--demand", regime,
                "--episodes", str(num_seeds),
                "--seed-start", str(seed_start),
                "--num-seconds", str(HPARAMS["num_seconds"]),
                "--checkpoint", str(best_ckpt),
                "--output-dir", str(slice_dir),
                "--resume",
                "--save-telemetry",
            ]
            _run_subprocess_with_retries(
                eval_cmd,
                model_prefix=model_variant,
                worker_log_path=worker_log,
                max_retries=max_retries,
            )
            _merge_and_summarize_regime(regime_dir)

        eval_wall_s = time.time() - t_eval
        _log(f"COMPLETED (train={train_wall_s:.1f}s, eval={eval_wall_s:.1f}s).", prefix=model_variant)
        return True
    except Exception as exc:
        status_label = "failed"
        error_msg = f"{type(exc).__name__}: {exc}"
        _log(f"[ERROR] Failed: {error_msg}", prefix=model_variant)
        return False
    finally:
        _update_status(status_label, err=error_msg)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the resumable mixed-demand training & 50-seed eval queue (sequential or parallel)."
    )
    parser.add_argument(
        "--network",
        type=str,
        default="resco_4x4",
        choices=NETWORK_CHOICES,
        help="Target road network (default: resco_4x4).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(PENDING_MODELS),
        choices=MODEL_VARIANTS,
        help="Models to execute (default: the 4 pending D3QN models only).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of models to train & evaluate concurrently (default: 4).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Automatic --resume retries per failed train/eval subprocess before marking a model failed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    network = args.network
    workers = max(1, int(args.workers))
    models = list(args.models)
    ordered_keys = [(network, m) for m in MODEL_VARIANTS]

    outputs_root = REPO_ROOT / "outputs"
    status_csv = outputs_root / "logs" / f"{network}_mixed_queue_status.csv"
    status_map = _load_existing_status(status_csv)

    _log(
        f"Starting local mixed queue for {network} (models={models}, workers={workers}, max_retries={args.max_retries}) on "
        f"{'CUDA: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
    )

    any_failed = False
    try:
        if workers == 1:
            for model_variant in models:
                ok = _run_single_model(
                    outputs_root=outputs_root,
                    network=network,
                    model_variant=model_variant,
                    status_csv=status_csv,
                    status_map=status_map,
                    ordered_keys=ordered_keys,
                    max_retries=args.max_retries,
                    stagger_s=0.0,
                )
                if not ok:
                    any_failed = True
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                future_map = {
                    pool.submit(
                        _run_single_model,
                        outputs_root=outputs_root,
                        network=network,
                        model_variant=model_variant,
                        status_csv=status_csv,
                        status_map=status_map,
                        ordered_keys=ordered_keys,
                        max_retries=args.max_retries,
                        stagger_s=idx * 3.0,
                    ): model_variant
                    for idx, model_variant in enumerate(models)
                }
                for fut in as_completed(future_map):
                    if not fut.result():
                        any_failed = True
    finally:
        # Clean up any orphaned SUMO processes only once all workers have exited
        subprocess.run(["pkill", "-x", "sumo"], check=False)

    _build_and_print_leaderboard(outputs_root, network, list(MODEL_VARIANTS))
    _log(f"Queue status persisted to: {status_csv}")
    if any_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
