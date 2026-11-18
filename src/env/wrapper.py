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
        validate_contracts: bool = True,
    ):
        super().__init__(env)
        self.cfg = cfg
        self.validate_contracts = validate_contracts
        self.delta_time_s = float(cfg["shield"]["delta_time_s"])  # 5.0
        self.norm_cfg = cfg["observation"]
        self.state_dim = int(self.norm_cfg["state_dim"])          # 33
        self.num_lanes = int(self.norm_cfg["num_lanes"])          # 8
        self.max_phases = int(self.norm_cfg["max_phases"])        # 8

        self.ts_id = self.unwrapped.ts_ids[0]
        self.ts = self.unwrapped.traffic_signals[self.ts_id]
        self.num_actions = self.ts.num_green_phases

        self.shield = SafetyShield(self.num_actions, cfg, enabled=use_shield)
        self.reward_calc = TrafficRewardCalculator(
            cfg, use_starvation_reward=use_starvation_reward
        )

        self.observation_space = gym.spaces.Box(
            low=0.0, high=10.0, shape=(self.state_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(self.num_actions)

        self.current_phase = 0
        self.elapsed_green_s = 0.0
        self._prev_lane_vehicles = {}

    def _get_traci(self):
        return self.unwrapped.sumo

    def _collect_lane_metrics(self):
        traci = self._get_traci()
        lanes = list(self.ts.lanes[: self.num_lanes])
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
        self, counts: np.ndarray, queues: np.ndarray, max_waits: np.ndarray
    ) -> np.ndarray:
        max_cap = float(self.norm_cfg["max_lane_capacity_veh"])
        max_wait_norm = float(self.norm_cfg["max_wait_norm_s"])
        max_green_norm = float(self.norm_cfg["max_green_norm_s"])

        norm_counts = np.clip(counts / max_cap, 0.0, 1.0)
        norm_queues = np.clip(queues / max_cap, 0.0, 1.0)

        phase_one_hot = np.zeros(self.max_phases, dtype=np.float32)
        phase_one_hot[self.current_phase] = 1.0

        norm_elapsed = np.array(
            [np.clip(self.elapsed_green_s / max_green_norm, 0.0, 1.0)], dtype=np.float32
        )
        norm_waits = np.clip(max_waits / max_wait_norm, 0.0, 1.0)

        return np.concatenate(
            [norm_counts, norm_queues, phase_one_hot, norm_elapsed, norm_waits]
        ).astype(np.float32)

    def _snapshot_lane_vehicles(self) -> dict[str, set[str]]:
        traci = self._get_traci()
        return {
            lane_id: set(traci.lane.getLastStepVehicleIDs(lane_id))
            for lane_id in self.ts.lanes[: self.num_lanes]
        }

    def _sim_seconds(self, duration_s: float) -> tuple[float, float]:
        traci = self._get_traci()
        step_len = traci.simulation.getDeltaT()
        num_substeps = max(1, int(round(duration_s / step_len)))
        co2_mg = 0.0
        dist_m = 0.0
        for _ in range(num_substeps):
            traci.simulationStep()
            for lane_id in self.ts.lanes[: self.num_lanes]:
                co2_mg += traci.lane.getCO2Emission(lane_id) * step_len
                for v in traci.lane.getLastStepVehicleIDs(lane_id):
                    dist_m += traci.vehicle.getSpeed(v) * step_len
        return co2_mg, dist_m

    def reset(self, *, seed=None, options=None):
        _, raw_info = self.env.reset(seed=seed, options=options)
        self.reward_calc.reset()
        self.current_phase = 0
        self.elapsed_green_s = 0.0

        traci = self._get_traci()
        green_state = self.ts.green_phases[self.current_phase].state
        traci.trafficlight.setRedYellowGreenState(self.ts_id, green_state)

        self._prev_lane_vehicles = self._snapshot_lane_vehicles()
        lanes, counts, queues, delays, max_waits = self._collect_lane_metrics()
        obs = self._build_observation(counts, queues, max_waits)
        mask = self.shield.get_action_mask(self.current_phase, self.elapsed_green_s)

        info = dict(raw_info)
        info.update(
            {
                "sim_time": float(traci.simulation.getTime()),
                "ts_id": self.ts_id,
                "active_phase": int(self.current_phase),
                "elapsed_green_s": float(self.elapsed_green_s),
                "proposed_action": int(self.current_phase),
                "executed_action": int(self.current_phase),
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
            validate_step_contract(obs, mask, info, self.num_actions)

        return obs, info

    def step(self, action: int):
        proposed_action = int(action)
        executed_action, mask_approved = self.shield.validate_action(
            proposed_action, self.current_phase, self.elapsed_green_s
        )
        phase_switched = executed_action != self.current_phase
        traci = self._get_traci()

        total_co2_mg = 0.0
        total_dist_m = 0.0

        if phase_switched:
            curr_state = self.ts.green_phases[self.current_phase].state
            next_state = self.ts.green_phases[executed_action].state

            yellow_state = self.shield.build_yellow_state(curr_state, next_state)
            traci.trafficlight.setRedYellowGreenState(self.ts_id, yellow_state)
            co2, dist = self._sim_seconds(self.shield.yellow_s)
            total_co2_mg += co2
            total_dist_m += dist

            all_red_state = self.shield.build_all_red_state(curr_state)
            traci.trafficlight.setRedYellowGreenState(self.ts_id, all_red_state)
            co2, dist = self._sim_seconds(self.shield.all_red_s)
            total_co2_mg += co2
            total_dist_m += dist

            self.current_phase = executed_action
            self.elapsed_green_s = 0.0
            traci.trafficlight.setRedYellowGreenState(self.ts_id, next_state)
        else:
            curr_state = self.ts.green_phases[self.current_phase].state
            traci.trafficlight.setRedYellowGreenState(self.ts_id, curr_state)
            co2, dist = self._sim_seconds(self.delta_time_s)
            total_co2_mg += co2
            total_dist_m += dist
            self.elapsed_green_s += self.delta_time_s

        curr_lane_vehicles = self._snapshot_lane_vehicles()
        throughput = sum(
            len(self._prev_lane_vehicles[l] - curr_lane_vehicles[l])
            for l in self._prev_lane_vehicles
        )
        self._prev_lane_vehicles = curr_lane_vehicles

        lanes, counts, queues, delays, max_waits = self._collect_lane_metrics()
        obs = self._build_observation(counts, queues, max_waits)
        mask = self.shield.get_action_mask(self.current_phase, self.elapsed_green_s)

        max_cap = float(self.norm_cfg["max_lane_capacity_veh"])
        norm_queues = np.clip(queues / max_cap, 0.0, 1.0)
        reward = self.reward_calc.compute(norm_queues, delays, max_waits, throughput)

        sim_time = float(traci.simulation.getTime())
        terminated = sim_time >= float(self.unwrapped.sim_max_time)
        truncated = False

        info = {
            "sim_time": sim_time,
            "ts_id": self.ts_id,
            "active_phase": int(self.current_phase),
            "elapsed_green_s": float(self.elapsed_green_s),
            "proposed_action": proposed_action,
            "executed_action": int(executed_action),
            "mask_approved": bool(mask_approved),
            "phase_switched": bool(phase_switched),
            "action_mask": mask,
            "lane_ids": lanes,
            "lane_vehicle_counts": counts,
            "lane_queues": queues,
            "lane_delays": delays,
            "lane_max_waits": max_waits,
            "step_throughput": int(throughput),
            "step_co2_mg": float(total_co2_mg),
            "step_distance_m": float(total_dist_m),
        }

        if self.validate_contracts:
            validate_step_contract(obs, mask, info, self.num_actions)

        return obs, float(reward), terminated, truncated, info