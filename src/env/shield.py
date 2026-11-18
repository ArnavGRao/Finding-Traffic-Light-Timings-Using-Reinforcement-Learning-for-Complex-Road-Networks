import numpy as np


class SafetyShield:
    def __init__(self, num_actions: int, cfg: dict, enabled: bool = True):
        s_cfg = cfg["shield"]
        self.num_actions = num_actions
        self.enabled = enabled
        self.min_green_s = float(s_cfg["min_green_s"])  # 10.0
        self.yellow_s = float(s_cfg["yellow_s"])        # 3.0
        self.all_red_s = float(s_cfg["all_red_s"])      # 2.0
        self.mask_allow = float(s_cfg["mask_allow"])    # 0.0
        self.mask_block = float(s_cfg["mask_block"])    # -1e9

    def get_action_mask(self, current_phase: int, elapsed_green_s: float) -> np.ndarray:
        mask = np.full(self.num_actions, self.mask_allow, dtype=np.float32)
        if not self.enabled:
            return mask

        if elapsed_green_s < self.min_green_s:
            mask[:] = self.mask_block
            mask[current_phase] = self.mask_allow
        return mask

    def validate_action(
        self, proposed_action: int, current_phase: int, elapsed_green_s: float
    ) -> tuple[int, bool]:
        if not self.enabled:
            return proposed_action, True

        if proposed_action != current_phase and elapsed_green_s < self.min_green_s:
            return current_phase, False
        return proposed_action, True

    @staticmethod
    def build_all_red_state(green_state: str) -> str:
        return "".join("r" if c in ("G", "g", "y", "Y") else c for c in green_state)

    @staticmethod
    def build_yellow_state(current_green_state: str, next_green_state: str) -> str:
        chars = []
        for curr_c, next_c in zip(current_green_state, next_green_state):
            if curr_c in ("G", "g") and next_c not in ("G", "g"):
                chars.append("y")
            else:
                chars.append(curr_c)
        return "".join(chars)