import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingDQN(nn.Module):
    """Actor (Policy) Model."""
    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=128):
        super(DuelingDQN, self).__init__()

        self.seed = torch.manual_seed(seed)

        self.feature = nn.Sequential(
            nn.Linear(state_size, fc1_units),
            nn.ReLU()
        )

        self.advantage = nn.Sequential(
            nn.Linear(fc1_units, fc2_units),
            nn.ReLU(),
            nn.Linear(fc2_units, action_size)
        )

        self.value = nn.Sequential(
            nn.Linear(fc1_units, fc2_units),
            nn.ReLU(),
            nn.Linear(fc2_units, 1)
        )

    def forward(self, x):
        x = self.feature(x)
        advantage = self.advantage(x)
        value     = self.value(x)
        return value + advantage  - advantage.mean()
