class MaxPressureController:
    """Adaptive controller selecting the green phase with maximum upstream-minus-downstream pressure."""

    def __init__(self, min_green_s: float = 10.0, use_halting: bool = True):
        self.min_green_s = float(min_green_s)
        self.use_halting = use_halting
        self._phase_links: dict[int, list[tuple[str, str]]] | None = None

    def _init_phase_links(self, env) -> None:
        unwrapped = env.unwrapped
        ts_id = unwrapped.ts_ids[0]
        ts = unwrapped.traffic_signals[ts_id]
        traci = unwrapped.sumo
        controlled_links = traci.trafficlight.getControlledLinks(ts_id)

        self._phase_links = {}
        for phase_idx, green_phase in enumerate(ts.green_phases):
            pairs = []
            for sig_char, link_group in zip(green_phase.state, controlled_links):
                if sig_char in ("G", "g") and link_group:
                    for in_lane, out_lane, _ in link_group:
                        pairs.append((in_lane, out_lane))
            self._phase_links[phase_idx] = pairs

    def _lane_load(self, traci, lane_id: str) -> float:
        if not lane_id:
            return 0.0
        if self.use_halting:
            return float(traci.lane.getLastStepHaltingNumber(lane_id))
        return float(traci.lane.getLastStepVehicleNumber(lane_id))

    def select_action(self, info: dict, env) -> int:
        if self._phase_links is None:
            self._init_phase_links(env)

        current_phase = int(info["active_phase"])
        elapsed_green_s = float(info["elapsed_green_s"])
        if elapsed_green_s < self.min_green_s:
            return current_phase

        traci = env.unwrapped.sumo
        best_phase = current_phase
        best_pressure = -float("inf")

        for phase_idx, pairs in self._phase_links.items():
            pressure = 0.0
            for in_lane, out_lane in pairs:
                pressure += self._lane_load(traci, in_lane) - self._lane_load(traci, out_lane)
            if pressure > best_pressure:
                best_pressure = pressure
                best_phase = phase_idx

        return best_phase