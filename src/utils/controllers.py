from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np
import sumo_rl
import torch

from src.baselines.fixed_time import FixedTimeController
from src.baselines.max_pressure import MaxPressureController
from src.baselines.vanilla_dqn import VanillaDQNAgent
from src.env.wrapper import SafetyShieldedTrafficEnv
from src.utils.contracts import OBS_DIM, load_default_config


class ControllerAdapter(Protocol):
    """Uniform controller interface shared by rollout scripts and live dashboard evaluation."""

    name: str

    def reset(self) -> None:
        ...

    def select_action(self, obs: np.ndarray, info: dict[str, Any], env: Any) -> int:
        ...


@dataclass(frozen=True)
class ControllerSpec:
    """Configuration metadata and default wrapper flags for a supported controller."""

    key: str
    display_name: str
    requires_checkpoint: bool
    default_use_shield: bool
    default_reward_mode: str
    is_d3qn: bool = False
    ablation_tag: str | None = None
    description: str = ""


@dataclass(frozen=True)
class ControllerAvailability:
    """Runtime availability status for a controller and any discovered .pt checkpoints."""

    spec: ControllerSpec
    code_available: bool
    checkpoint_files: tuple[Path, ...]
    ready: bool
    reason: str


CONTROLLER_SPECS: dict[str, ControllerSpec] = {
    "fixed_time": ControllerSpec(
        key="fixed_time",
        display_name="Fixed-Time (30s)",
        requires_checkpoint=False,
        default_use_shield=True,
        default_reward_mode="starvation",
        description="Pre-timed 30s cyclic signal phase baseline.",
    ),
    "max_pressure": ControllerSpec(
        key="max_pressure",
        display_name="Max-Pressure",
        requires_checkpoint=False,
        default_use_shield=True,
        default_reward_mode="starvation",
        description="Adaptive upstream-minus-downstream queue pressure controller.",
    ),
    "vanilla_dqn": ControllerSpec(
        key="vanilla_dqn",
        display_name="Vanilla DQN",
        requires_checkpoint=True,
        default_use_shield=False,
        default_reward_mode="queue_diff",
        description="Unshielded single-stream DQN baseline optimizing queue difference.",
    ),
    "d3qn_a": ControllerSpec(
        key="d3qn_a",
        display_name="Model A (Baseline D3QN)",
        requires_checkpoint=True,
        default_use_shield=False,
        default_reward_mode="delay_diff",
        is_d3qn=True,
        ablation_tag="A",
        description="Dueling Double DQN without safety shield or starvation penalty.",
    ),
    "d3qn_b": ControllerSpec(
        key="d3qn_b",
        display_name="Model B (D3QN + Shield)",
        requires_checkpoint=True,
        default_use_shield=True,
        default_reward_mode="delay_diff",
        is_d3qn=True,
        ablation_tag="B",
        description="Dueling Double DQN with action-mask safety shield.",
    ),
    "d3qn_c": ControllerSpec(
        key="d3qn_c",
        display_name="Model C (D3QN + Starvation)",
        requires_checkpoint=True,
        default_use_shield=False,
        default_reward_mode="starvation",
        is_d3qn=True,
        ablation_tag="C",
        description="Dueling Double DQN with starvation-aware reward (no shield).",
    ),
    "d3qn_d": ControllerSpec(
        key="d3qn_d",
        display_name="Model D (Complete Proposed)",
        requires_checkpoint=True,
        default_use_shield=True,
        default_reward_mode="starvation",
        is_d3qn=True,
        ablation_tag="D",
        description="Complete D3QN with both safety shield and starvation-aware reward.",
    ),
}

_CONTROLLER_ALIASES: dict[str, str] = {
    "fixed": "fixed_time",
    "fixed-time": "fixed_time",
    "fixedtime": "fixed_time",
    "max-pressure": "max_pressure",
    "maxpressure": "max_pressure",
    "dqn": "vanilla_dqn",
    "vanilla-dqn": "vanilla_dqn",
    "vanilladqn": "vanilla_dqn",
    "model_a": "d3qn_a",
    "model-a": "d3qn_a",
    "d3qn-a": "d3qn_a",
    "a": "d3qn_a",
    "model_b": "d3qn_b",
    "model-b": "d3qn_b",
    "d3qn-b": "d3qn_b",
    "b": "d3qn_b",
    "model_c": "d3qn_c",
    "model-c": "d3qn_c",
    "d3qn-c": "d3qn_c",
    "c": "d3qn_c",
    "model_d": "d3qn_d",
    "model-d": "d3qn_d",
    "d3qn-d": "d3qn_d",
    "d": "d3qn_d",
}

D3QNLoaderFn = Callable[
    [Path, int, dict[str, Any], ControllerSpec],
    ControllerAdapter,
]
_CUSTOM_D3QN_LOADER: D3QNLoaderFn | None = None


def register_d3qn_loader(loader_fn: D3QNLoaderFn | None) -> None:
    """Register (or clear with None) a custom D3QN adapter factory hook."""
    global _CUSTOM_D3QN_LOADER
    _CUSTOM_D3QN_LOADER = loader_fn


def resolve_controller_key(name: str) -> str:
    """Resolve controller key or alias into canonical key in CONTROLLER_SPECS."""
    normalized = name.strip().lower()
    canonical = _CONTROLLER_ALIASES.get(normalized, normalized)
    if canonical not in CONTROLLER_SPECS:
        valid = ", ".join(CONTROLLER_SPECS.keys())
        raise ValueError(f"Unknown controller '{name}'. Valid options: {valid}")
    return canonical


class FixedTimeAdapter:
    """Adapter wrapping FixedTimeController."""

    def __init__(self, num_actions: int, green_duration_s: float = 30.0):
        self.name = f"FixedTime{int(round(green_duration_s))}s"
        self.num_actions = num_actions
        self.green_duration_s = float(green_duration_s)
        self._ctrl = FixedTimeController(
            num_actions=num_actions, green_duration_s=self.green_duration_s
        )

    def reset(self) -> None:
        self._ctrl = FixedTimeController(
            num_actions=self.num_actions, green_duration_s=self.green_duration_s
        )

    def select_action(self, obs: np.ndarray, info: dict[str, Any], env: Any) -> int:
        return int(self._ctrl.select_action(info))


class MaxPressureAdapter:
    """Adapter wrapping MaxPressureController."""

    def __init__(self, min_green_s: float = 10.0, use_halting: bool = True):
        self.name = "MaxPressure"
        self.min_green_s = float(min_green_s)
        self.use_halting = bool(use_halting)
        self._ctrl = MaxPressureController(
            min_green_s=self.min_green_s, use_halting=self.use_halting
        )

    def reset(self) -> None:
        self._ctrl = MaxPressureController(
            min_green_s=self.min_green_s, use_halting=self.use_halting
        )

    def select_action(self, obs: np.ndarray, info: dict[str, Any], env: Any) -> int:
        return int(self._ctrl.select_action(info, env))


class VanillaDQNAdapter:
    """Adapter wrapping VanillaDQNAgent for deterministic checkpoint evaluation."""

    def __init__(
        self,
        num_actions: int,
        checkpoint_path: str | Path | None = None,
        state_dim: int = OBS_DIM,
        device: str | None = None,
    ):
        self.name = "VanillaDQN"
        self.agent = VanillaDQNAgent(
            state_dim=state_dim,
            num_actions=num_actions,
            device=device,
        )
        if checkpoint_path is not None:
            self._load_checkpoint(Path(checkpoint_path))
        self.agent.online_net.eval()

    def _load_checkpoint(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Vanilla DQN checkpoint not found: {path}")
        try:
            self.agent.load(str(path))
            return
        except KeyError:
            pass

        ckpt = torch.load(str(path), map_location=self.agent.device)
        if isinstance(ckpt, dict):
            for key in ("online_net", "q_network", "model_state_dict", "state_dict"):
                if key in ckpt:
                    self.agent.online_net.load_state_dict(ckpt[key])
                    self.agent.target_net.load_state_dict(self.agent.online_net.state_dict())
                    return
        self.agent.online_net.load_state_dict(ckpt)
        self.agent.target_net.load_state_dict(self.agent.online_net.state_dict())

    def reset(self) -> None:
        self.agent.online_net.eval()

    def select_action(self, obs: np.ndarray, info: dict[str, Any], env: Any) -> int:
        return int(self.agent.select_action(obs, deterministic=True))


class D3QNControllerAdapter:
    """Adapter wrapping dynamically discovered D3QNAgent implementations."""

    def __init__(self, agent: Any, display_name: str):
        self.agent = agent
        self.name = display_name
        if hasattr(self.agent, "eval"):
            self.agent.eval()
        online = getattr(self.agent, "online_net", None)
        if online is not None and hasattr(online, "eval"):
            online.eval()

    def reset(self) -> None:
        if hasattr(self.agent, "reset") and callable(self.agent.reset):
            self.agent.reset()

    def select_action(self, obs: np.ndarray, info: dict[str, Any], env: Any) -> int:
        mask = info.get("action_mask")
        fn = getattr(self.agent, "select_action", None) or getattr(self.agent, "act", None)
        if fn is None:
            raise AttributeError("D3QNAgent must expose a select_action(...) or act(...) method.")

        sig = inspect.signature(fn)
        params = sig.parameters
        kwargs: dict[str, Any] = {}
        if "action_mask" in params:
            kwargs["action_mask"] = mask
        elif "mask" in params:
            kwargs["mask"] = mask
        if "deterministic" in params:
            kwargs["deterministic"] = True
        elif "eval_mode" in params:
            kwargs["eval_mode"] = True
        elif "epsilon" in params:
            kwargs["epsilon"] = 0.0

        positional = [
            p
            for p in params.values()
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            and p.name not in kwargs
        ]
        if len(positional) >= 2 and mask is not None and "mask" in positional[1].name.lower():
            return int(fn(obs, mask, **kwargs))
        return int(fn(obs, **kwargs))


def detect_d3qn_agent_class() -> tuple[type[Any] | None, str | None]:
    """Safely inspect src.agents for a D3QNAgent class without raising on missing modules."""
    candidate_modules = [
        "src.agents",
        "src.agents.d3qn",
        "src.agents.d3qn_agent",
        "src.agents.agent",
    ]
    agents_dir = Path("src/agents")
    if agents_dir.exists():
        for py_file in sorted(agents_dir.glob("*.py")):
            if py_file.name != "__init__.py":
                mod_name = f"src.agents.{py_file.stem}"
                if mod_name not in candidate_modules:
                    candidate_modules.append(mod_name)

    candidate_class_names = ("D3QNAgent", "DuelingDQNAgent", "DuelingDoubleDQNAgent")

    for mod_name in candidate_modules:
        try:
            module = importlib.import_module(mod_name)
        except Exception:
            continue

        for cls_name in candidate_class_names:
            cls = getattr(module, cls_name, None)
            if inspect.isclass(cls):
                return cls, mod_name

        for attr_name, obj in vars(module).items():
            if inspect.isclass(obj) and "d3qn" in attr_name.lower():
                return obj, mod_name

    return None, None


def _instantiate_d3qn_from_checkpoint(
    agent_cls: type[Any],
    checkpoint_path: Path,
    num_actions: int,
    cfg: dict[str, Any],
    spec: ControllerSpec,
) -> ControllerAdapter:
    """Instantiate a discovered D3QNAgent class and load checkpoint weights."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"D3QN checkpoint not found: {checkpoint_path}")

    if hasattr(agent_cls, "from_checkpoint") and callable(agent_cls.from_checkpoint):
        agent = agent_cls.from_checkpoint(str(checkpoint_path))
        return D3QNControllerAdapter(agent, spec.display_name)

    init_params = inspect.signature(agent_cls.__init__).parameters
    init_kwargs: dict[str, Any] = {}
    if "state_dim" in init_params:
        init_kwargs["state_dim"] = OBS_DIM
    if "obs_dim" in init_params:
        init_kwargs["obs_dim"] = OBS_DIM
    if "num_actions" in init_params:
        init_kwargs["num_actions"] = num_actions
    if "action_dim" in init_params:
        init_kwargs["action_dim"] = num_actions
    if "cfg" in init_params:
        init_kwargs["cfg"] = cfg
    if "config" in init_params:
        init_kwargs["config"] = cfg
    if "use_shield" in init_params:
        init_kwargs["use_shield"] = spec.default_use_shield

    agent = agent_cls(**init_kwargs)
    if hasattr(agent, "load") and callable(agent.load):
        agent.load(str(checkpoint_path))
    else:
        device = getattr(agent, "device", "cpu")
        ckpt = torch.load(str(checkpoint_path), map_location=device)
        net = getattr(agent, "online_net", None) or getattr(agent, "q_network", None)
        if net is None:
            raise AttributeError("Unable to load checkpoint into D3QNAgent (no load() or online_net).")
        state_dict = ckpt.get("online_net", ckpt.get("model_state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
        net.load_state_dict(state_dict)

    return D3QNControllerAdapter(agent, spec.display_name)


def discover_checkpoints(
    search_dirs: tuple[str | Path, ...] = ("outputs", "checkpoints", "models"),
) -> list[Path]:
    """Discover saved PyTorch (.pt/.pth) checkpoint files on disk."""
    found: list[Path] = []
    seen: set[Path] = set()
    for root_dir in search_dirs:
        root = Path(root_dir)
        if not root.exists():
            continue
        for pattern in ("*.pt", "*.pth"):
            for path in sorted(root.rglob(pattern)):
                resolved = path.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    found.append(path)
    return found


def _checkpoint_matches_spec(path: Path, spec: ControllerSpec) -> bool:
    """Heuristically match a checkpoint path to a controller specification."""
    stem = path.as_posix().lower()
    if spec.key == "vanilla_dqn":
        return "vanilla" in stem or "vdqn" in stem or ("dqn" in stem and "d3qn" not in stem)
    if spec.ablation_tag:
        tag = spec.ablation_tag.lower()
        tokens = (f"model_{tag}", f"model-{tag}", f"d3qn_{tag}", f"d3qn-{tag}", f"ablation_{tag}")
        return any(tok in stem for tok in tokens)
    return False


def get_controller_availabilities(
    search_dirs: tuple[str | Path, ...] = ("outputs", "checkpoints", "models"),
) -> dict[str, ControllerAvailability]:
    """Check runtime availability for all baseline and D3QN ablation controllers."""
    all_checkpoints = discover_checkpoints(search_dirs=search_dirs)
    d3qn_cls, d3qn_module = detect_d3qn_agent_class()

    availabilities: dict[str, ControllerAvailability] = {}
    for key, spec in CONTROLLER_SPECS.items():
        if not spec.is_d3qn:
            matched = tuple(p for p in all_checkpoints if _checkpoint_matches_spec(p, spec))
            if not spec.requires_checkpoint:
                availabilities[key] = ControllerAvailability(
                    spec=spec,
                    code_available=True,
                    checkpoint_files=(),
                    ready=True,
                    reason="Built-in deterministic baseline ready.",
                )
            else:
                reason = (
                    f"{len(matched)} matching checkpoint(s) found."
                    if matched
                    else "VanillaDQNAgent ready (provide or select a .pt checkpoint path)."
                )
                availabilities[key] = ControllerAvailability(
                    spec=spec,
                    code_available=True,
                    checkpoint_files=matched,
                    ready=True,
                    reason=reason,
                )
            continue

        matched_d3qn = tuple(p for p in all_checkpoints if _checkpoint_matches_spec(p, spec))
        code_ok = _CUSTOM_D3QN_LOADER is not None or d3qn_cls is not None
        if not code_ok:
            reason = (
                "Unavailable: D3QNAgent class not yet present in src/agents/"
                + (
                    f" ({len(matched_d3qn)} checkpoint(s) found)."
                    if matched_d3qn
                    else " and no .pt checkpoints found on disk."
                )
            )
            ready = False
        elif not matched_d3qn:
            reason = (
                f"D3QNAgent detected in {d3qn_module}; no Model {spec.ablation_tag} .pt checkpoint auto-detected (enter custom path)."
            )
            ready = False
        else:
            reason = (
                f"D3QNAgent detected in {d3qn_module}; {len(matched_d3qn)} checkpoint(s) found."
            )
            ready = True

        availabilities[key] = ControllerAvailability(
            spec=spec,
            code_available=code_ok,
            checkpoint_files=matched_d3qn,
            ready=ready,
            reason=reason,
        )

    return availabilities


def build_controller(
    controller_key: str,
    num_actions: int,
    cfg: dict[str, Any] | None = None,
    checkpoint_path: str | Path | None = None,
    green_duration_s: float | None = None,
    allow_untrained: bool = False,
) -> ControllerAdapter:
    """Instantiate a baseline or checkpoint-backed controller adapter."""
    resolved_key = resolve_controller_key(controller_key)
    spec = CONTROLLER_SPECS[resolved_key]
    cfg = cfg or load_default_config()

    if resolved_key == "fixed_time":
        green_s = (
            float(green_duration_s)
            if green_duration_s is not None
            else float(cfg["evaluation"]["fixed_time_green_s"])
        )
        return FixedTimeAdapter(num_actions=num_actions, green_duration_s=green_s)

    if resolved_key == "max_pressure":
        return MaxPressureAdapter(min_green_s=float(cfg["shield"]["min_green_s"]))

    if resolved_key == "vanilla_dqn":
        if checkpoint_path is None and not allow_untrained:
            raise ValueError(
                "Vanilla DQN requires a --checkpoint .pt path (or allow_untrained=True)."
            )
        return VanillaDQNAdapter(
            num_actions=num_actions,
            checkpoint_path=checkpoint_path,
            state_dim=int(cfg["observation"]["state_dim"]),
        )

    if checkpoint_path is None:
        raise ValueError(f"{spec.display_name} requires a trained .pt checkpoint path.")

    ckpt = Path(checkpoint_path)
    if _CUSTOM_D3QN_LOADER is not None:
        return _CUSTOM_D3QN_LOADER(ckpt, num_actions, cfg, spec)

    d3qn_cls, _ = detect_d3qn_agent_class()
    if d3qn_cls is None:
        raise RuntimeError(
            f"Cannot load {spec.display_name}: D3QNAgent class is not yet implemented in src/agents/."
        )

    return _instantiate_d3qn_from_checkpoint(d3qn_cls, ckpt, num_actions, cfg, spec)


def infer_network_and_demand(net_file: str | Path, route_file: str | Path) -> tuple[str, str]:
    """Infer clean network and demand-regime labels from SUMO net and route file paths."""
    net_path = Path(net_file)
    route_path = Path(route_file)
    network_name = net_path.parent.name if net_path.parent.name else net_path.stem

    net_prefix = net_path.name.split(".net.")[0]
    route_stem = route_path.name.split(".rou.")[0]
    if route_stem.startswith(net_prefix + "-"):
        demand_regime = route_stem[len(net_prefix) + 1 :]
    elif "-" in route_stem:
        demand_regime = route_stem.rsplit("-", 1)[-1]
    else:
        demand_regime = route_stem
    return network_name, demand_regime


def discover_networks_and_routes(
    networks_dir: str | Path = "networks",
) -> tuple[list[Path], list[Path]]:
    """Discover SUMO .net.xml and .rou.xml files under networks/."""
    root = Path(networks_dir)
    if not root.exists():
        return [], []
    net_files = sorted(root.rglob("*.net.xml"))
    route_files = sorted(root.rglob("*.rou.xml"))
    return net_files, route_files


def make_traffic_env(
    cfg: dict[str, Any],
    net_file: str | Path,
    route_file: str | Path,
    use_shield: bool,
    reward_mode: str,
    num_seconds: int = 3600,
    delta_time_s: int | None = None,
    validate_contracts: bool = True,
) -> SafetyShieldedTrafficEnv:
    """Create a headless SafetyShieldedTrafficEnv configured with matching shield/reward flags."""
    dt = int(delta_time_s if delta_time_s is not None else cfg["shield"]["delta_time_s"])
    base_env = sumo_rl.SumoEnvironment(
        net_file=str(net_file),
        route_file=str(route_file),
        single_agent=True,
        use_gui=False,
        num_seconds=int(num_seconds),
        delta_time=dt,
        yellow_time=0,
        min_green=0,
        sumo_warnings=False,
        additional_sumo_cmd="--no-step-log",
    )
    return SafetyShieldedTrafficEnv(
        base_env,
        cfg=cfg,
        use_shield=bool(use_shield),
        reward_mode=str(reward_mode),
        validate_contracts=validate_contracts,
    )
