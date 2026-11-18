import numpy as np


class SumTree:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.tree = np.zeros(2 * self.capacity - 1, dtype=np.float64)
        self.write_idx = 0
        self.size = 0

    def total(self) -> float:
        return float(self.tree[0])

    def add(self, priority: float) -> int:
        data_idx = self.write_idx
        tree_idx = data_idx + self.capacity - 1
        self.update(tree_idx, priority)

        self.write_idx = (self.write_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return data_idx

    def update(self, tree_idx: int, priority: float) -> None:
        delta = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += delta

    def get(self, cumulative_value: float) -> tuple[int, float, int]:
        idx = 0
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                break
            if cumulative_value <= self.tree[left]:
                idx = left
            else:
                cumulative_value -= self.tree[left]
                idx = right

        data_idx = idx - (self.capacity - 1)
        return idx, float(self.tree[idx]), data_idx


class PrioritizedReplayBuffer:
    def __init__(
        self,
        capacity: int = 100_000,
        state_dim: int = 33,
        num_actions: int = 4,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100_000,
        eps: float = 1e-6,
    ):
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.beta_start = float(beta_start)
        self.beta_frames = max(1, int(beta_frames))
        self.eps = float(eps)
        self.frame = 0

        self.tree = SumTree(self.capacity)
        self.states = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.next_states = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.masks = np.zeros((self.capacity, num_actions), dtype=np.float32)
        self.next_masks = np.zeros((self.capacity, num_actions), dtype=np.float32)

        self.max_priority = 1.0

    def __len__(self) -> int:
        return self.tree.size

    def current_beta(self) -> float:
        fraction = min(1.0, self.frame / self.beta_frames)
        return self.beta_start + fraction * (1.0 - self.beta_start)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        mask: np.ndarray,
        next_mask: np.ndarray,
    ) -> None:
        priority = self.max_priority ** self.alpha
        data_idx = self.tree.add(priority)

        self.states[data_idx] = state
        self.actions[data_idx] = int(action)
        self.rewards[data_idx] = float(reward)
        self.next_states[data_idx] = next_state
        self.dones[data_idx] = float(done)
        self.masks[data_idx] = mask
        self.next_masks[data_idx] = next_mask

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        if len(self) < batch_size:
            raise ValueError(f"Cannot sample {batch_size} items from buffer of size {len(self)}")

        self.frame += 1
        beta = self.current_beta()
        total_p = self.tree.total()
        segment = total_p / batch_size

        tree_indices = np.zeros(batch_size, dtype=np.int64)
        data_indices = np.zeros(batch_size, dtype=np.int64)
        priorities = np.zeros(batch_size, dtype=np.float64)

        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            val = np.random.uniform(low, min(high, total_p - 1e-8))
            t_idx, prio, d_idx = self.tree.get(val)
            # Guard against rounding into unpopulated tail slots
            d_idx = min(max(d_idx, 0), len(self) - 1)
            tree_indices[i] = t_idx
            priorities[i] = max(prio, self.eps)
            data_indices[i] = d_idx

        probs = np.clip(priorities / max(total_p, self.eps), 1e-12, None)
        weights = (len(self) * probs) ** (-beta)
        weights = (weights / weights.max()).astype(np.float32)

        return {
            "states": self.states[data_indices],
            "actions": self.actions[data_indices],
            "rewards": self.rewards[data_indices],
            "next_states": self.next_states[data_indices],
            "dones": self.dones[data_indices],
            "masks": self.masks[data_indices],
            "next_masks": self.next_masks[data_indices],
            "tree_indices": tree_indices,
            "is_weights": weights,
        }

    def update_priorities(self, tree_indices: np.ndarray, td_errors: np.ndarray) -> None:
        abs_errors = np.abs(td_errors) + self.eps
        for t_idx, err in zip(tree_indices, abs_errors):
            clipped = float(err)
            self.max_priority = max(self.max_priority, clipped)
            self.tree.update(int(t_idx), clipped ** self.alpha)