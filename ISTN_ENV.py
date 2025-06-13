import numpy as np
import random

class ISTNEnv:
    def __init__(self, num_satellites, num_ground_stations, max_buffer=10, max_energy=1.0, max_time=100, seed=0):
        self.num_satellites = num_satellites
        self.num_ground_stations = num_ground_stations
        self.max_buffer = max_buffer
        self.max_energy = max_energy
        self.max_time = max_time
        self.random = random.Random(seed)
        self.n_agents = num_satellites + num_ground_stations
        # 邻接表在每个time slot动态传入
        self.neighbors = None
        # 每个agent本地状态维度
        self.obs_dim = 3
        # 动作维度在动态拓扑下等于邻居数量，这里给出可能的上界
        self.action_dim = num_satellites + num_ground_stations
        self.reset()

    def _build_neighbors(self):
        # 邻接表，简单：卫星和卫星4连通，地面站与所有卫星连通
        neighbors = {}
        for i in range(self.num_satellites):
            # 环状连接+自己前后两个（可自定义为实际拓扑）
            nbs = [(i-1)%self.num_satellites, (i+1)%self.num_satellites]
            # 地面站全部可连
            nbs += [self.num_satellites + j for j in range(self.num_ground_stations)]
            neighbors[i] = nbs
        for j in range(self.num_ground_stations):
            # 地面站只连所有卫星
            neighbors[self.num_satellites + j] = [i for i in range(self.num_satellites)]
        return neighbors

    def initialize_satellites(self):
        satellites = {}
        for i in range(self.num_satellites):
            satellites[i] = {
                'energy': self.random.uniform(0.5, self.max_energy), # 初始化高一点
                'buffer': self.random.randint(0, self.max_buffer//2),
                'latency': self.random.uniform(1, 10)
            }
        return satellites

    def initialize_ground_stations(self):
        ground_stations = {}
        for i in range(self.num_ground_stations):
            ground_stations[i] = {
                'energy': self.random.uniform(0.5, self.max_energy),
                'buffer': self.random.randint(0, self.max_buffer//2),
                'latency': self.random.uniform(1, 10)
            }
        return ground_stations

    def get_obs(self, neighbors):
        """Return observations for all agents.

        Each agent observes its own state concatenated with the state of its
        current neighbors.
        """
        obs = []
        for i in range(self.n_agents):
            if i < self.num_satellites:
                own = self.satellites[i]
            else:
                own = self.ground_stations[i - self.num_satellites]

            state = [own['energy'], own['buffer'], own['latency']]
            for nb in neighbors.get(i, []):
                if nb < self.num_satellites:
                    nb_node = self.satellites[nb]
                else:
                    nb_node = self.ground_stations[nb - self.num_satellites]
                state.extend([nb_node['energy'], nb_node['buffer'], nb_node['latency']])
            obs.append(np.array(state, dtype=np.float32))

        return obs

    def step(self, actions, neighbors):
        """One environment step.

        Parameters
        ----------
        actions : list
            length ``n_agents``. ``actions[i]`` is the chosen neighbor index for
            agent ``i``.
        neighbors : dict
            Mapping from agent index to a list of its current neighbors.
        """
        self.neighbors = neighbors
        cost_energy = np.zeros(self.n_agents)
        cost_loss = 0
        rewards = np.zeros(self.n_agents)
        # 实际routing和buffer转移
        for idx, action in enumerate(actions):
            # 当前节点的buffer有包才能转发
            if idx < self.num_satellites:
                node = self.satellites[idx]
            else:
                node = self.ground_stations[idx - self.num_satellites]

            if node['buffer'] > 0:
                # 检查动作是否有效（在邻居范围内）
                if idx not in neighbors or action >= len(neighbors[idx]):
                    # 动作无效，直接跳过或给予惩罚
                    rewards[idx] -= 0.5  # 无效动作惩罚
                    continue

                # 目标必须是邻居
                target = neighbors[idx][action]
                # 目标节点
                if target < self.num_satellites:
                    tgt_node = self.satellites[target]
                else:
                    tgt_node = self.ground_stations[target - self.num_satellites]
                # 缓冲是否满
                if tgt_node['buffer'] < self.max_buffer:
                    tgt_node['buffer'] += 1
                    node['buffer'] -= 1
                    # 模拟奖励
                    rewards[idx] += 1
                else:
                    # 丢包
                    cost_loss += 1
                    node['buffer'] -= 1
                    rewards[idx] -= 1
                # 能耗简单加一
                node['energy'] -= 0.01
                cost_energy[idx] += 0.01
        self.time_slot += 1
        done = self.time_slot >= self.max_time
        obs = self.get_obs(neighbors)
        info = {}
        costs = {'energy': cost_energy, 'loss': cost_loss}
        return obs, rewards, done, costs, info

    def reset(self):
        self.satellites = self.initialize_satellites()
        self.ground_stations = self.initialize_ground_stations()
        self.time_slot = 0
        # 初始拓扑随机生成一次
        self.neighbors = self._build_neighbors()
        return self.get_obs(self.neighbors)

# 测试代码
if __name__ == '__main__':
    env = ISTNEnv(num_satellites=3, num_ground_stations=2)
    obs = env.reset()
    print("init_obs:", obs)
    for step in range(5):
        # 动态生成邻接关系
        neighbors = env._build_neighbors()
        actions = [env.random.choice(range(len(neighbors[i]))) for i in range(env.n_agents)]
        obs, rewards, done, costs, info = env.step(actions, neighbors)
        print(f"step={step+1}, obs={obs}, rewards={rewards}, costs={costs}")
        if done:
            break
