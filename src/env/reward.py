import numpy as np


class TrafficRewardCalculator:
    def __init__(self, cfg: dict, use_starvation_reward: bool = True):
        r_cfg = cfg["reward"]
        self.use_starvation = use_starvation_reward
        self.w1 = float(r_cfg["w1_queue"])          # 0.25
        self.w2 = float(r_cfg["w2_delay"])          # 0.25
        self.w3 = float(r_cfg["w3_starvation"])     # 0.35
        self.w4 = float(r_cfg["w4_throughput"])     # 0.15
        self.w_thresh = float(r_cfg["w_thresh_s"])  # 60.0
        self.alpha = float(r_cfg["alpha"])          # 2.0
        self.t_max = float(r_cfg["t_max_per_step"]) # 10.0
        self._prev_total_delay = None

    def reset(self):
        self._prev_total_delay = None

    def compute(
        self,
        norm_queues: np.ndarray,
        lane_delays_s: np.ndarray,
        lane_max_waits_s: np.ndarray,
        step_throughput: float,
    ) -> float:
        total_delay = float(np.sum(lane_delays_s))

        if not self.use_starvation:
            if self._prev_total_delay is None:
                self._prev_total_delay = total_delay
                return 0.0
            reward = (self._prev_total_delay - total_delay) / self.w_thresh
            self._prev_total_delay = total_delay
            return float(reward)

        queue_term = self.w1 * float(np.sum(norm_queues))
        delay_term = self.w2 * float(np.sum(lane_delays_s / self.w_thresh))
        starvation_term = self.w3 * float(
            np.sum((lane_max_waits_s / self.w_thresh) ** self.alpha)
        )
        throughput_term = self.w4 * (float(step_throughput) / self.t_max)

        return -(queue_term + delay_term + starvation_term) + throughput_term