import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.agents.networks import DuelingQNetwork, VanillaQNetwork
from src.agents.replay_buffer import PrioritizedReplayBuffer


class D3QNAgent:
    def __init__(
        self,
        state_dim: int = 33,
        num_actions: int = 4,
        hidden_dim: int = 128,
        lr: float = 5e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        batch_size: int = 64,
        buffer_capacity: int = 100_000,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_frames: int = 100_000,
        dueling: bool = True,
        double_dqn: bool = True,
        grad_clip_norm: float = 10.0,
        device: str | None = None,
    ):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.batch_size = int(batch_size)
        self.dueling = bool(dueling)
        self.double_dqn = bool(double_dqn)
        self.grad_clip_norm = float(grad_clip_norm)

        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        net_cls = DuelingQNetwork if self.dueling else VanillaQNetwork
        self.online_net = net_cls(state_dim, num_actions, hidden_dim).to(self.device)
        self.target_net = copy.deepcopy(self.online_net).to(self.device)
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss(reduction="none")

        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=buffer_capacity,
            state_dim=state_dim,
            num_actions=num_actions,
            alpha=per_alpha,
            beta_start=per_beta_start,
            beta_frames=per_beta_frames,
        )
        self.train_steps = 0

    @torch.no_grad()
    def act(
        self, state: np.ndarray, action_mask: np.ndarray, epsilon: float = 0.0
    ) -> int:
        valid_actions = np.where(action_mask > -1e8)[0]
        if len(valid_actions) == 0:
            valid_actions = np.arange(self.num_actions)

        if epsilon > 0.0 and np.random.rand() < epsilon:
            return int(np.random.choice(valid_actions))

        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = torch.as_tensor(action_mask, dtype=torch.float32, device=self.device).unsqueeze(0)

        q_values = self.online_net(state_t)
        masked_q = q_values + mask_t
        return int(torch.argmax(masked_q, dim=-1).item())

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        mask: np.ndarray,
        next_mask: np.ndarray,
    ) -> None:
        self.replay_buffer.add(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            mask=mask,
            next_mask=next_mask,
        )

    def compute_target_and_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        next_masks: torch.Tensor,
        is_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        current_q = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.double_dqn:
                online_next_q = self.online_net(next_states)
                masked_online_next_q = online_next_q + next_masks
                best_next_actions = masked_online_next_q.argmax(dim=-1, keepdim=True)
                target_next_q = (
                    self.target_net(next_states).gather(1, best_next_actions).squeeze(1)
                )
            else:
                masked_target_next_q = self.target_net(next_states) + next_masks
                target_next_q = masked_target_next_q.max(dim=-1).values

            y_t = rewards + self.gamma * (1.0 - dones) * target_next_q

        td_errors = y_t - current_q
        elementwise_loss = self.loss_fn(current_q, y_t)
        weighted_loss = (elementwise_loss * is_weights).mean()
        return weighted_loss, td_errors.detach(), y_t.detach()

    def soft_update_target(self) -> None:
        for target_param, online_param in zip(
            self.target_net.parameters(), self.online_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * online_param.data + (1.0 - self.tau) * target_param.data
            )

    def train_step(self) -> dict[str, float] | None:
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size)
        states = torch.as_tensor(batch["states"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.int64, device=self.device)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(batch["next_states"], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device)
        next_masks = torch.as_tensor(batch["next_masks"], dtype=torch.float32, device=self.device)
        is_weights = torch.as_tensor(batch["is_weights"], dtype=torch.float32, device=self.device)

        loss, td_errors, y_t = self.compute_target_and_loss(
            states, actions, rewards, next_states, dones, next_masks, is_weights
        )

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.online_net.parameters(), self.grad_clip_norm
        )
        self.optimizer.step()

        self.replay_buffer.update_priorities(
            batch["tree_indices"], td_errors.cpu().numpy()
        )
        self.soft_update_target()
        self.train_steps += 1

        return {
            "loss": float(loss.item()),
            "mean_td_error": float(td_errors.abs().mean().item()),
            "mean_target_q": float(y_t.mean().item()),
            "grad_norm": float(grad_norm.item()),
            "per_beta": float(self.replay_buffer.current_beta()),
        }

    def save(self, filepath: str) -> None:
        torch.save(
            {
                "online_net": self.online_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "train_steps": self.train_steps,
                "dueling": self.dueling,
                "double_dqn": self.double_dqn,
            },
            filepath,
        )

    def load(self, filepath: str) -> None:
        ckpt = torch.load(filepath, map_location=self.device, weights_only=True)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.train_steps = int(ckpt["train_steps"])