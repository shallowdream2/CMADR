import torch
import torch.nn as nn
from torch.nn import functional as F
from AC import ActorNet, CriticNet  # 假设AC.py中定义了ActorNet和CriticNet

class MultiAgentSystem:
    def __init__(self, n_agents, obs_dim, action_dim, hidden_dim, device='cpu'):
        self.n_agents = n_agents
        self.device = device
        # 每个agent自己的actor/critic
        self.actors = [ActorNet(obs_dim, hidden_dim, action_dim).to(device) for _ in range(n_agents)]
        self.critics = [CriticNet(obs_dim, hidden_dim).to(device) for _ in range(n_agents)]
        # 全局actor/critic
        # self.global_actor = ActorNet(obs_dim * n_agents, hidden_dim, action_dim * n_agents).to(device)
        self.global_critic = CriticNet(obs_dim * n_agents, hidden_dim).to(device)
        # 优化器（可单独/合并设计）
        # self.optim_actors = [torch.optim.Adam(actor.parameters(), lr=1e-3) for actor in self.actors]
        self.optim_critics = [torch.optim.Adam(critic.parameters(), lr=1e-3) for critic in self.critics]
        # self.optim_global_actor = torch.optim.Adam(self.global_actor.parameters(), lr=1e-3)
        self.optim_global_critic = torch.optim.Adam(self.global_critic.parameters(), lr=1e-3)

    def select_actions(self, obs_n):
        # obs_n: [n_agents, obs_dim]，返回每个agent的动作
        actions = []
        for i in range(self.n_agents):
            obs = torch.tensor(obs_n[i], dtype=torch.float32, device=self.device).unsqueeze(0)
            probs = self.actors[i](obs)
            m = torch.distributions.Categorical(probs)
            action = m.sample().item()
            actions.append(action)
        return actions

    def evaluate_global(self, obs_n):
        # 拼成 [1, n_agents*obs_dim]
        obs_cat = torch.tensor(obs_n, dtype=torch.float32, device=self.device).view(1, -1)
        value = self.global_critic(obs_cat)
        # 可输出全局action概率等
        return value
