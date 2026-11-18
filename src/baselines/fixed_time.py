class FixedTimeController:
    """Pre-timed cyclic controller with fixed green phase duration (default 30s)."""

    def __init__(self, num_actions: int, green_duration_s: float = 30.0):
        self.num_actions = num_actions
        self.green_duration_s = float(green_duration_s)

    def select_action(self, info: dict) -> int:
        current_phase = int(info["active_phase"])
        elapsed_green_s = float(info["elapsed_green_s"])
        if elapsed_green_s >= self.green_duration_s:
            return (current_phase + 1) % self.num_actions
        return current_phase