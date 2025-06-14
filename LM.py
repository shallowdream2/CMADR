# todo: 加入loss lagrange乘子对loss的贡献再训练

import torch
import torch.nn.functional as F
import numpy as np

class LagrangeMultiplier:
    """ 单约束拉格朗日乘子 """
    def __init__(self, init_value=1.0, lr=0.01, min_val=1e-3, max_val=100.0, device='cpu'):
        self.value = torch.tensor([init_value], dtype=torch.float32, requires_grad=True, device=device)
        self.lr = lr
        self.min_val = min_val
        self.max_val = max_val

    def update(self, cost_violation):
        # cost_violation: torch scalar，>0 表示违反，<0表示未违反
        grad = cost_violation.detach()
        with torch.no_grad():
            self.value += self.lr * grad
            self.value.clamp_(self.min_val, self.max_val)
        self.value.requires_grad = True

    def __call__(self):
        return self.value

# === 训练主循环 ===
def train_cmadr(env, mac, num_episodes=500, gamma=0.98, cost_limits=None, device='cpu'):
    """
    env: ISTNEnv
    mac: MultiAgentSystem
    cost_limits: dict, e.g. {'energy': 0.5, 'loss': 5}
    """
    n_agents = mac.n_agents
    cost_limits = cost_limits or {'energy': 0.5, 'loss': 5}

    # 1. 初始化拉格朗日乘子
    lagrange_energy = LagrangeMultiplier(init_value=1.0, lr=0.01, device=device)
    lagrange_loss = LagrangeMultiplier(init_value=1.0, lr=0.01, device=device)

    for ep in range(num_episodes):
        obs = env.reset()
        episode_transitions = []
        done = False
        ep_reward = 0
        ep_energy_cost = 0
        ep_loss_cost = 0
        step_count = 0
        while not done:
            # 动态生成当前时隙的拓扑
            neighbors = env._build_neighbors()
            actions = mac.select_actions(obs,neighbors)
            next_obs, rewards, done, costs, info = env.step(actions, neighbors)
                
            # 存储一条transition
            global_obs = np.concatenate(obs, axis=0) # shape = [n_agents * obs_dim]
            transition = {
                'obs': obs, 
                'global_obs': global_obs,  # 新增
                'actions': actions,
                'rewards': rewards,
                'cost_energy': costs['energy'],
                'cost_loss': costs['loss'],
                'next_obs': next_obs, 
                'done': done
            }
            episode_transitions.append(transition)
            obs = next_obs
            step_count += 1
            ep_reward += np.sum(rewards)
            ep_energy_cost += np.sum(costs['energy'])
            ep_loss_cost += costs['loss']

        # 2. 转换采样数据为批量
        obs_batch = torch.tensor(np.array([tr['obs'] for tr in episode_transitions]), dtype=torch.float32, device=device)  # [T, n_agents, obs_dim]
        act_batch = torch.tensor(np.array([tr['actions'] for tr in episode_transitions]), dtype=torch.long, device=device) # [T, n_agents]
        rew_batch = torch.tensor(np.array([tr['rewards'] for tr in episode_transitions]), dtype=torch.float32, device=device) # [T, n_agents]
        cost_energy_batch = torch.tensor(np.array([tr['cost_energy'] for tr in episode_transitions]), dtype=torch.float32, device=device)
        cost_loss_batch = torch.tensor(np.array([tr['cost_loss'] for tr in episode_transitions]), dtype=torch.float32, device=device)
        next_obs_batch = torch.tensor(np.array([tr['next_obs'] for tr in episode_transitions]), dtype=torch.float32, device=device)
        done_batch = torch.tensor(np.array([tr['done'] for tr in episode_transitions]), dtype=torch.float32, device=device)
        global_obs_batch = torch.tensor(np.array([tr['global_obs'] for tr in episode_transitions]), dtype=torch.float32, device=device) # [T, n_agents * obs_dim]

        # 3. 计算advantage/target（简单时序差分或GAE均可，这里用TD）
        # 对每个agent分别更新
        for agent_idx in range(n_agents):
            agent = mac.actors[agent_idx]
            critic = mac.critics[agent_idx]
            optimizer_a = mac.optim_actors[agent_idx]
            optimizer_c = mac.optim_critics[agent_idx]
            obs_agent = obs_batch[:, agent_idx, :]       # [T, obs_dim]
            act_agent = act_batch[:, agent_idx]          # [T]
            rew_agent = rew_batch[:, agent_idx]          # [T]
            cost_e_agent = cost_energy_batch[:, agent_idx] # [T]

            # 计算值函数和目标 - 基于reward而不是cost
            values = critic(obs_agent).squeeze(-1)       # [T]
            # bootstrapped TD target - 使用reward
            td_target = rew_agent + gamma * torch.cat([values[1:], values[-1:]]) * (1-done_batch)
            advantage = td_target[:-1] - values[:-1]

            # Actor loss: 最大化reward (策略梯度)
            logits = agent(obs_agent[:-1])
            logp = torch.log(logits.gather(1, act_agent[:-1].unsqueeze(-1)).squeeze(-1) + 1e-8)
            actor_loss = -torch.mean(logp * advantage.detach())  # 现在使用基于reward的advantage
            
            # Critic loss: 预测reward的价值
            critic_loss = F.mse_loss(values[:-1], td_target[:-1].detach())

        # 全局cost处理（约束项）
        global_values = mac.global_critic(global_obs_batch).squeeze(-1) # [T]

        global_cost_enegy = cost_energy_batch.mean(dim=1)  # [T]
        global_cost_enegy_to_go = []
        running = 0
        for t in reversed(range(len(global_cost_enegy))):
            running = global_cost_enegy[t] + gamma * running * (1 - done_batch[t])
            global_cost_enegy_to_go.insert(0, running)
        global_cost_enegy_to_go = torch.tensor(global_cost_enegy_to_go, dtype=torch.float32, device=device)

        global_cost_loss = cost_loss_batch  # [T]
        global_cost_loss_to_go = []
        running = 0
        for t in reversed(range(len(global_cost_loss))):
            running = global_cost_loss[t] + gamma * running * (1 - done_batch[t])
            global_cost_loss_to_go.insert(0, running)
        global_cost_loss_to_go = torch.tensor(global_cost_loss_to_go, dtype=torch.float32, device=device)
        
    
        
        # 全局critic损失：预测cost
        global_critic_loss = F.mse_loss(global_values[:-1], global_cost_enegy_to_go[:-1].detach())

        # Lagrange约束项：惩罚cost超出限制
        cost_violation = global_cost_enegy_to_go[:-1].mean() - cost_limits['energy']
        lagrange_enegy_term = lagrange_energy() * cost_violation

        # loss cost violation
        cost_violation_loss = global_cost_loss_to_go[:-1].mean() - cost_limits['loss']
        lagrange_loss_term = lagrange_loss() * cost_violation_loss

        # 总损失：最大化reward + 约束cost
        total_loss = actor_loss + critic_loss + global_critic_loss + lagrange_loss_term+ lagrange_enegy_term

        optimizer_a.zero_grad()
        optimizer_c.zero_grad()
        mac.optim_global_critic.zero_grad()
        total_loss.backward()
        optimizer_a.step()
        optimizer_c.step()
        mac.optim_global_critic.step()

        lagrange_energy.update(cost_violation)
        lagrange_loss.update(cost_violation_loss)


        # 4. 全局网络（可选：辅助优化/target value）
        # joint_obs = obs_batch.reshape(obs_batch.shape[0], -1)  # [T, n_agents*obs_dim]
        # mac.global_critic(joint_obs)  # ...

        if ep % 10 == 0:
            print(f"\nEpisode {ep}: reward={ep_reward:.2f} energy={ep_energy_cost:.2f} loss={ep_loss_cost:.2f} λ_e={lagrange_energy().item():.2f}\n")

