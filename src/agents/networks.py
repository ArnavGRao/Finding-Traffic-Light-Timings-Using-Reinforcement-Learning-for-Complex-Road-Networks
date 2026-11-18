import torch
import torch.nn as nn


class VanillaQNetwork(nn.Module):
    def __init__(self, state_dim: int = 33, num_actions: int = 4, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim: int = 33, num_actions: int = 4, hidden_dim: int = 128):
        super().__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions),
        )

    def forward_streams(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_layer(state)
        value = self.value_stream(features)          # (batch, 1)
        advantage = self.advantage_stream(features)  # (batch, num_actions)
        return value, advantage

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        value, advantage = self.forward_streams(state)
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        return q_values