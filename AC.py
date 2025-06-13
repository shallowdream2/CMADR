import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorNet(nn.Module):
    def __init__(self, obs_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.softmax(self.fc2(x), dim=-1)
        return x

class CriticNet(nn.Module):
    def __init__(self, obs_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        value = self.fc2(x)
        return value
