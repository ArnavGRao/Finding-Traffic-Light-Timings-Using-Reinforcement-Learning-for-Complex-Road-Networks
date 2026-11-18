from collections.abc import Callable, Mapping
import gymnasium as gym
import numpy as np
from src.env.reward import TrafficRewardCalculator
from src.env.shield import SafetyShield
from src.utils.contracts import validate_step_contract


class SafetyShieldedTrafficEnv(gym.Wrapper):
    def __init__(
        self,
        env,
        cfg: dict,
        use_shield: bool = True,
        use_starvation_reward: bool = True,
        reward_mode: str | None = None,
        validate_contracts: bool = True,
        multi_agent: bool | None = None,
        substep_callback: Callable[[], None] | None = None,
    ):
        super().__init__(env)
        self.cfg = cfg
        self.validate_contracts = validate_contracts
        self.substep_callback = substep_callback
        self.delta_time_s = float(cfg["shield"]["delta_time_s"])  # 5.0
        self.norm_cfg = cfg["observation"]
        self.state_dim = int(self.norm_cfg["state_dim"])          # 33
        self.num_lanes = int(self.norm_cfg["num_lanes"])          # 8
        self.max_phases = int(self.norm_cfg["max_phases"])        # 8

        self.ts_ids = list(self.unwrapped.ts_ids)
        if multi_agent is None:
            self.is_multi_agent = (
                not bool(getattr(self.unwrapped, "single_agent", True))
            ) or len(self.ts_ids) > 1
        else:
            self.is_multi_agent = bool(multi_agent)

        self.traffic_signals = {
            tid: self.unwrapped.traffic_signals[tid] for tid in self.ts_ids
        }
        self.num_actions_map = {
            tid: int(self.traffic_signals[tid].num_green_phases)
            for tid in self.ts_ids
        }
        self.shields = {
            tid: SafetyShield(self.num_actions_map[tid], cfg, enabled=use_shield)
            for tid in self.ts_ids
        }
        self.reward_calcs = {
            tid: TrafficRewardCalculator(
                cfg,
                use_starvation_reward=use_starvation_reward,
                reward_mode=reward_mode,
            )
            for tid in self.ts_ids
        }

        # Single-agent compatibility attributes
        self.ts_id = self.ts_ids[0]
        self.ts = self.traffic_signals[self.ts_id]
        self.num_actions = self.num_actions_map[self.ts_id]
        self.shield = self.shields[self.ts_id]
        self.reward_calc = self.reward_calcs[self.ts_id]

        self.current_phases: dict[str, int] = {tid: 0 for tid in self.ts_ids}
        self.elapsed_greens: dict[str, float] = {tid: 0.0 for tid in self.ts_ids}
        self._prev_lane_vehicles: dict[str, dict[str, set[str]]] = {}

    def set_substep_callback(self, callback: Callable[[], None] | None) -> None:
        """Register or clear a callback invoked after every SUMO simulationStep."""
        self.substep_callback = callback

    def _get_traci(self):
        return self.unwrapped.sumo

    def _collect_lane_metrics(self, ts_id: str | None = None):
        tid = self.ts_id if ts_id is None else ts_id
        traci = self._get_traci()
        lanes = list(self.traffic_signals[tid].lanes[: self.num_lanes])
        counts = np.zeros(self.num_lanes, dtype=np.float32)
        queues = np.zeros(self.num_lanes, dtype=np.float32)
        delays = np.zeros(self.num_lanes, dtype=np.float32)
        max_waits = np.zeros(self.num_lanes, dtype=np.float32)

        for idx, lane_id in enumerate(lanes):
            veh_ids = traci.lane.getLastStepVehicleIDs(lane_id)
            counts[idx] = float(len(veh_ids))
            queues[idx] = float(traci.lane.getLastStepHaltingNumber(lane_id))
            if veh_ids:
                waits = [traci.vehicle.getWaitingTime(v) for v in veh_ids]
                max_waits[idx] = float(max(waits))
                delays[idx] = float(np.mean(waits))

        return lanes, counts, queues, delays, max_waits

    def _build_observation(
        self,
        counts: np.ndarray,
        queues: np.ndarray,
        max_waits: np.ndarray,
        ts_id: str | None = None,
    ) -> np.ndarray:
        tid = self.ts_id if ts_id is None else ts_id
        max_cap = float(self.norm_cfg["max_lane_capacity_veh"])
        max_wait_norm = float(self.norm_cfg["max_wait_norm_s"])
        max_green_norm = float(self.norm_cfg["max_green_norm_s"])

        norm_counts = np.clip(counts / max_cap, 0.0, 1.0)
        norm_queues = np.clip(queues / max_cap, 0.0, 1.0)

        phase_one_hot = np.zeros(self.max_phases, dtype=np.float32)
        phase_one_hot[self.current_phases[tid]] = 1.0

        norm_elapsed = np.array(
            [np.clip(self.elapsed_greens[tid] / max_green_norm, 0.0, 1.0)],
            dtype=np.float32,
        )
        norm_waits = np.clip(max_waits / max_wait_norm, 0.0, 1.0)

        return np.concatenate(
            [norm_counts, norm_queues, phase_one_hot, norm_elapsed, norm_waits]
        ).astype(np.float32)

    def _snapshot_lane_vehicles(self, ts_id: str | None = None) -> dict[str, set[str]]:
        tid = self.ts_id if ts_id is None else ts_id
        traci = self._get_traci()
        return {
            lane_id: set(traci.lane.getLastStepVehicleIDs(lane_id))
            for lane_id in self.traffic_signals[tid].lanes[: self.num_lanes]
        }

    def _step_simulation_substeps(
        self, phase_plans: dict[str, dict]
    ) -> tuple[dict[str, float], dict[str, float]]:
        traci = self._get_traci()
        step_len = float(traci.simulation.getDeltaT())
        num_substeps = max(1, int(round(self.delta_time_s / step_len)))
        co2_by_ts = {tid: 0.0 for tid in self.ts_ids}
        dist_by_ts = {tid: 0.0 for tid in self.ts_ids}

        for sub_idx in range(num_substeps):
            elapsed_in_step = sub_idx * step_len
            for tid, plan in phase_plans.items():
                if not plan["switched"]:
                    state_str = plan["curr_state"]
                elif elapsed_in_step < plan["yellow_s"]:
                    state_str = plan["yellow_state"]
                elif elapsed_in_step < plan["yellow_s"] + plan["all_red_s"]:
                    state_str = plan["all_red_state"]
                else:
                    state_str = plan["next_state"]
                traci.trafficlight.setRedYellowGreenState(tid, state_str)

            traci.simulationStep()
            if self.substep_callback is not None:
                self.substep_callback()

            for tid in self.ts_ids:
                for lane_id in self.traffic_signals[tid].lanes[: self.num_lanes]:
                    co2_by_ts[tid] += traci.lane.getCO2Emission(lane_id) * step_len
                    for v in traci.lane.getLastStepVehicleIDs(lane_id):
                        dist_by_ts[tid] += traci.vehicle.getSpeed(v) * step_len

        return co2_by_ts, dist_by_ts

    def reset(self, *, seed=None, options=None):
        reset_out = self.env.reset(seed=seed, options=options)
        if isinstance(reset_out, tuple) and len(reset_out) == 2:
            _, raw_info = reset_out
        else:
            raw_info = {}
        traci = self._get_traci()
        sim_time = float(traci.simulation.getTime())

        obs_dict: dict[str, np.ndarray] = {}
        info_dict: dict[str, dict] = {}
        base_info = dict(raw_info) if isinstance(raw_info, Mapping) else {}

        for tid in self.ts_ids:
            self.reward_calcs[tid].reset()
            self.current_phases[tid] = 0
            self.elapsed_greens[tid] = 0.0

            green_state = self.traffic_signals[tid].green_phases[0].state
            traci.trafficlight.setRedYellowGreenState(tid, green_state)

            self._prev_lane_vehicles[tid] = self._snapshot_lane_vehicles(tid)
            lanes, counts, queues, delays, max_waits = self._collect_lane_metrics(tid)
            obs = self._build_observation(counts, queues, max_waits, ts_id=tid)
            mask = self.shields[tid].get_action_mask(0, 0.0)

            info = dict(base_info) if not self.is_multi_agent else {}
            info.update(
                {
                    "sim_time": sim_time,
                    "ts_id": tid,
                    "active_phase": 0,
                    "elapsed_green_s": 0.0,
                    "proposed_action": 0,
                    "executed_action": 0,
                    "mask_approved": True,
                    "phase_switched": False,
                    "action_mask": mask,
                    "lane_ids": lanes,
                    "lane_vehicle_counts": counts,
                    "lane_queues": queues,
                    "lane_delays": delays,
                    "lane_max_waits": max_waits,
                    "step_throughput": 0,
                    "step_co2_mg": 0.0,
                    "step_distance_m": 0.0,
                }
            )

            if self.validate_contracts:
                validate_step_contract(obs, mask, info, self.num_actions_map[tid])

            obs_dict[tid] = obs
            info_dict[tid] = info

        self.current_phase = self.current_phases[self.ts_id]
        self.elapsed_green_s = self.elapsed_greens[self.ts_id]

        if self.is_multi_agent:
            return obs_dict, info_dict
        return obs_dict[self.ts_id], info_dict[self.ts_id]

    def step(self, action: int | Mapping[str, int]):
        if self.is_multi_agent:
            if not isinstance(action, Mapping):
                raise TypeError("Multi-agent SafetyShieldedTrafficEnv.step expects a mapping of {ts_id: action}.")
            action_map = {tid: int(action[tid]) for tid in self.ts_ids}
        else:
            action_map = {self.ts_id: int(action)}

        traci = self._get_traci()
        phase_plans: dict[str, dict] = {}
        step_decisions: dict[str, dict] = {}

        for tid in self.ts_ids:
            curr_phase = self.current_phases[tid]
            elapsed_green = self.elapsed_greens[tid]
            proposed_action = action_map[tid]
            shield = self.shields[tid]
            executed_action, mask_approved = shield.validate_action(
                proposed_action, curr_phase, elapsed_green
            )
            phase_switched = executed_action != curr_phase
            green_phases = self.traffic_signals[tid].green_phases
            curr_state = green_phases[curr_phase].state
            next_state = green_phases[executed_action].state

            phase_plans[tid] = {
                "switched": phase_switched,
                "curr_state": curr_state,
                "next_state": next_state,
                "yellow_state": shield.build_yellow_state(curr_state, next_state) if phase_switched else curr_state,
                "all_red_state": shield.build_all_red_state(curr_state) if phase_switched else curr_state,
                "yellow_s": shield.yellow_s,
                "all_red_s": shield.all_red_s,
            }
            step_decisions[tid] = {
                "proposed_action": proposed_action,
                "executed_action": int(executed_action),
                "mask_approved": bool(mask_approved),
                "phase_switched": bool(phase_switched),
                "next_state": next_state,
            }

        co2_by_ts, dist_by_ts = self._step_simulation_substeps(phase_plans)
        sim_time = float(traci.simulation.getTime())
        terminated = sim_time >= float(self.unwrapped.sim_max_time)
        truncated = False
        max_cap = float(self.norm_cfg["max_lane_capacity_veh"])

        obs_dict: dict[str, np.ndarray] = {}
        reward_dict: dict[str, float] = {}
        info_dict: dict[str, dict] = {}

        for tid in self.ts_ids:
            dec = step_decisions[tid]
            if dec["phase_switched"]:
                self.current_phases[tid] = dec["executed_action"]
                self.elapsed_greens[tid] = 0.0
                traci.trafficlight.setRedYellowGreenState(tid, dec["next_state"])
            else:
                self.elapsed_greens[tid] += self.delta_time_s

            curr_lane_vehicles = self._snapshot_lane_vehicles(tid)
            throughput = sum(
                len(self._prev_lane_vehicles[tid][lane_id] - curr_lane_vehicles[lane_id])
                for lane_id in self._prev_lane_vehicles[tid]
            )
            self._prev_lane_vehicles[tid] = curr_lane_vehicles

            lanes, counts, queues, delays, max_waits = self._collect_lane_metrics(tid)
            obs = self._build_observation(counts, queues, max_waits, ts_id=tid)
            mask = self.shields[tid].get_action_mask(
                self.current_phases[tid], self.elapsed_greens[tid]
            )

            norm_queues = np.clip(queues / max_cap, 0.0, 1.0)
            reward = float(
                self.reward_calcs[tid].compute(norm_queues, delays, max_waits, throughput)
            )

            info = {
                "sim_time": sim_time,
                "ts_id": tid,
                "active_phase": int(self.current_phases[tid]),
                "elapsed_green_s": float(self.elapsed_greens[tid]),
                "proposed_action": dec["proposed_action"],
                "executed_action": dec["executed_action"],
                "mask_approved": dec["mask_approved"],
                "phase_switched": dec["phase_switched"],
                "action_mask": mask,
                "lane_ids": lanes,
                "lane_vehicle_counts": counts,
                "lane_queues": queues,
                "lane_delays": delays,
                "lane_max_waits": max_waits,
                "step_throughput": int(throughput),
                "step_co2_mg": float(co2_by_ts[tid]),
                "step_distance_m": float(dist_by_ts[tid]),
            }

            if self.validate_contracts:
                validate_step_contract(obs, mask, info, self.num_actions_map[tid])

            obs_dict[tid] = obs
            reward_dict[tid] = reward
            info_dict[tid] = info

        self.current_phase = self.current_phases[self.ts_id]
        self.elapsed_green_s = self.elapsed_greens[self.ts_id]

        if self.is_multi_agent:
            terminated_dict = {tid: terminated for tid in self.ts_ids}
            terminated_dict["__all__"] = terminated
            truncated_dict = {tid: truncated for tid in self.ts_ids}
            truncated_dict["__all__"] = truncated
            return obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict

        return (
            obs_dict[self.ts_id],
            reward_dict[self.ts_id],
            terminated,
            truncated,
            info_dict[self.ts_id],
        )