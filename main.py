from ISTN_ENV import ISTNEnv
from MASys import MultiAgentSystem
from LM import train_cmadr

# Initialize environment, models, and trainer
env = ISTNEnv(num_satellites=5, num_ground_stations=5)
mac = MultiAgentSystem(n_agents=env.num_satellites, obs_dim=env.obs_dim, action_dim=env.action_dim, hidden_dim=64, device='cpu')

train_cmadr(env, mac, num_episodes=1000, gamma=0.98, cost_limits={'energy': 0.5, 'loss': 5}, device='cpu')