import numpy as np
import random

class ISTNEnv:
    def __init__(
        self,
        num_satellites,
        num_ground_stations,
        max_buffer=10,
        max_energy=1.0,
        max_time=100,
        seed=0,
        sat_positions=None,
        gs_positions=None,
        queries=None,
        sat_positions_per_slot=None,
        conn_threshold=30,
        neighbors_per_slot=None
    ):
        """ISTN 环境

        当 ``sat_positions`` 或 ``gs_positions`` 提供时，将使用给定的位置数据，否
        则随机生成。 ``queries`` 用于在 ``reset`` 时初始化地面站 buffer，方便在训
        练和预测阶段保持一致的数据输入。
        """
        self.num_satellites = num_satellites
        self.num_ground_stations = num_ground_stations
        self.max_buffer = max_buffer
        self.max_energy = max_energy
        self.max_time = max_time
        self.random = random.Random(seed)
        self.n_agents = num_satellites + num_ground_stations

        self.sat_positions_per_slot = sat_positions_per_slot

        if sat_positions_per_slot is not None:
            self.sat_positions = [np.array(p) for p in sat_positions_per_slot[0]]
        else:
            self.sat_positions = sat_positions or [
                np.array([random.uniform(0, 100), random.uniform(0, 100)])
                for _ in range(self.num_satellites)
            ]
        self.gs_positions = gs_positions or [
            np.array([random.uniform(0, 100), random.uniform(0, 100)])
            for _ in range(self.num_ground_stations)
        ]

        self.conn_threshold = conn_threshold

        # sort queries by time slot; each query should have {'src','dst','time'}
        self.queries = sorted(queries or [], key=lambda q: q.get('time', 0))
        self.query_index = 0  # pointer to next query to release

        # 读取邻接表
        self.neighbors_per_slot = neighbors_per_slot or []
        # 每个agent本地状态维度（自身+邻居+目的地距离）
        max_possible_neighbors = num_satellites + num_ground_stations - 1
        self.obs_dim = 3 + 3 * max_possible_neighbors + 1   # 末尾+1为“到目的地的距离”
        self.action_dim = num_satellites + num_ground_stations
        self.reset()

    def _build_neighbors(self):
        """Build neighbors based on current node positions."""
        neighbors = [[] for _ in range(self.n_agents)]

        # satellite-satellite links
        for i in range(self.num_satellites):
            for j in range(self.num_satellites):
                if i == j:
                    continue
                dist = np.linalg.norm(np.array(self.sat_positions[i]) - np.array(self.sat_positions[j]))
                if dist <= self.conn_threshold:
                    neighbors[i].append(j)

        # satellite-ground links
        for gs in range(self.num_ground_stations):
            gs_pos = self.gs_positions[gs]
            for sat in range(self.num_satellites):
                dist = np.linalg.norm(np.array(self.sat_positions[sat]) - np.array(gs_pos))
                if dist <= self.conn_threshold:
                    neighbors[sat].append(self.num_satellites + gs)
                    neighbors[self.num_satellites + gs].append(sat)

        # sort neighbor lists
        for i in range(self.n_agents):
            neighbors[i].sort()
        
        return neighbors
    
    def load_neighbors_per_slot(self, now_slot):
        """
        加载邻接表数据，适用于动态拓扑。
        :param now_slot: 当前时隙
        :return: 返回当前时隙的邻居关系列表
        """
        if self.neighbors_per_slot and now_slot < len(self.neighbors_per_slot):
            slot_data = self.neighbors_per_slot[now_slot]
            if isinstance(slot_data, dict) and 'neighbors' in slot_data:
                return slot_data['neighbors']
            else:
                return slot_data
        else:
            # 如果没有预定义的邻居关系，使用动态生成的方法
            return self._build_neighbors()
        

    def initialize_satellites(self):
        satellites = {}
        for i in range(self.num_satellites):
            satellites[i] = {
                'energy': self.random.uniform(0.5, self.max_energy),
                'buffer': [],  # buffer现在存储包对象
                'latency': self.random.uniform(1, 10)
            }
        return satellites

    def initialize_ground_stations(self):
        ground_stations = {}
        for i in range(self.num_ground_stations):
            ground_stations[i] = {
                'energy': self.random.uniform(0.5, self.max_energy),
                'buffer': [],
                'latency': self.random.uniform(1, 10)
            }
        return ground_stations

    def _create_packet(self):
        # 创建包：带目的地（地面站ID），初始来源地面站随机
        src = self.random.randint(0, self.num_ground_stations - 1)
        dst = self.random.randint(0, self.num_ground_stations - 1)
        while dst == src:
            dst = self.random.randint(0, self.num_ground_stations - 1)
        return {'dst': dst, 'hop': 0, 'src': src, 'path': [], 'start_time': self.time_slot}

    def _create_packet_from_query(self, src, dst):
        """根据给定的查询生成数据包"""
        return {
            'dst': dst,
            'hop': 0,
            'src': src,
            'path': [],
            'start_time': self.time_slot,
        }

    def _release_queries(self):
        """Release queries scheduled for the current time slot."""
        while self.query_index < len(self.queries) and \
                self.queries[self.query_index].get('time', 0) == self.time_slot:
            q = self.queries[self.query_index]
            gs = self.ground_stations[q['src']]
            if len(gs['buffer']) < self.max_buffer:
                pkt = self._create_packet_from_query(q['src'], q['dst'])
                gs['buffer'].append(pkt)
            self.query_index += 1

    def get_obs(self, neighbors):
        """
        每个agent观测：自身 + 所有邻居 + 到目的地的距离（取buffer中第一个包作为“当前处理包”，没有就为0）
        """
        obs = []
        max_possible_neighbors = self.num_satellites + self.num_ground_stations - 1
        for i in range(self.n_agents):
            # 节点自身
            if i < self.num_satellites:
                own = self.satellites[i]
            else:
                own = self.ground_stations[i - self.num_satellites]
            state = [own['energy'], len(own['buffer']), own['latency']]
            # 邻居状态
            current_neighbors = neighbors[i] if i < len(neighbors) else []
            for j in range(max_possible_neighbors):
                if j < len(current_neighbors):
                    nb = current_neighbors[j]
                    if nb < self.num_satellites:
                        nb_node = self.satellites[nb]
                    else:
                        nb_node = self.ground_stations[nb - self.num_satellites]
                    state.extend([nb_node['energy'], len(nb_node['buffer']), nb_node['latency']])
                else:
                    state.extend([0.0, 0.0, 0.0])
            # === 增加到目的地的距离 ===
            buf = own['buffer']
            if buf:
                dst_gs = buf[0]['dst']  # 取第一个包
                if i < self.num_satellites:
                    pos = self.sat_positions[i]
                else:
                    pos = self.gs_positions[i - self.num_satellites]
                dst_pos = self.gs_positions[dst_gs]
                dist = np.linalg.norm(np.array(pos) - np.array(dst_pos))
            else:
                dist = 0.0
            state.append(dist)
            obs.append(np.array(state, dtype=np.float32))
        return obs


    def step(self, actions, neighbors):
        """
        actions: list, 每个agent选一个邻居，将其buffer头部包转发过去（有包才转）
        """
        cost_energy = np.zeros(self.n_agents)
        cost_loss = 0
        delivered_packets = []
        transit_packets = []
        rewards = np.zeros(self.n_agents)
        
        # 奖励设计参数
        DELIVERY_REWARD = 10.0          # 成功交付基础奖励
        FORWARD_REWARD = 0.2            # 转发基础奖励
        DROP_PENALTY = -5.0             # 丢包惩罚
        INVALID_ACTION_PENALTY = -0.5   # 无效动作惩罚
        IDLE_PENALTY = -0.1             # 空闲惩罚
        ENERGY_COST_WEIGHT = 2.0        # 能耗权重
        DISTANCE_REWARD_WEIGHT = 0.1    # 距离奖励权重
        DELAY_PENALTY_WEIGHT = 0.05     # 时延惩罚权重
        
        
        # 转发
        for idx, action in enumerate(actions):
            # 确定当前节点 - 修复索引问题
            if idx < self.num_satellites:
                node = self.satellites[idx]
                node_type = "SAT"
            else:
                gs_idx = idx - self.num_satellites  
                if gs_idx < len(self.ground_stations):
                    node = self.ground_stations[gs_idx]
                    node_type = f"GS{gs_idx}"
                else:
                    print(f"Error: Invalid agent index {idx}")
                    rewards[idx] += INVALID_ACTION_PENALTY
                    continue

            if node['buffer']:
                pkt = node['buffer'][0]
                dst_gs = pkt['dst']
                
               
                # 检查动作有效性
                current_neighbors = neighbors[idx] if idx < len(neighbors) else []
                if action >= len(current_neighbors):
                    # if self.time_slot < 3:
                    #     print(f"        Invalid action: {action} >= {len(current_neighbors)}")
                    rewards[idx] += INVALID_ACTION_PENALTY
                    continue
                    
                target = current_neighbors[action]
                
                # 确定目标节点
                if target < self.num_satellites:
                    tgt_node = self.satellites[target]
                    tgt_type = f"SAT{target}"
                else:
                    tgt_gs_idx = target - self.num_satellites
                    if tgt_gs_idx < len(self.ground_stations):
                        tgt_node = self.ground_stations[tgt_gs_idx]
                        tgt_type = f"GS{tgt_gs_idx}"
                    else:
                        rewards[idx] += INVALID_ACTION_PENALTY
                        continue
                    
                # 缓冲是否满
                if len(tgt_node['buffer']) < self.max_buffer:
                    # 路径+1跳
                    pkt = node['buffer'].pop(0)
                    pkt['hop'] += 1
                    pkt['path'].append(target)
                    tgt_node['buffer'].append(pkt)
                    
                    # if self.time_slot < 3:
                    #     print(f"        SUCCESS! Forwarded from {node_type} to {tgt_type}")
                    
                    # 计算距离奖励（朝向目的地的移动给予奖励）
                    current_pos = self.sat_positions[idx] if idx < self.num_satellites else self.gs_positions[idx - self.num_satellites]
                    target_pos = self.sat_positions[target] if target < self.num_satellites else self.gs_positions[target - self.num_satellites]
                    dst_pos = self.gs_positions[dst_gs]
                    
                    # 计算当前节点和目标节点到目的地的距离
                    current_to_dst = np.linalg.norm(np.array(current_pos) - np.array(dst_pos))
                    target_to_dst = np.linalg.norm(np.array(target_pos) - np.array(dst_pos))
                    
                    # 如果转发使包更接近目的地，给予距离奖励
                    distance_improvement = current_to_dst - target_to_dst
                    distance_reward = distance_improvement * DISTANCE_REWARD_WEIGHT
                    
                    # 计算时延惩罚
                    packet_age = self.time_slot - pkt['start_time']
                    delay_penalty = packet_age * DELAY_PENALTY_WEIGHT
                    
                    # 若目标是地面站且正好为目的地，则交付
                    if (target >= self.num_satellites) and ((target - self.num_satellites) == dst_gs):
                        delivered_packets.append(pkt)
                        tgt_node['buffer'].pop()  # 交付出队
                        
                        # 交付奖励：基础奖励 + 距离奖励 - 时延惩罚 - 跳数惩罚
                        hop_penalty = pkt['hop'] * 0.1  # 跳数惩罚
                        delivery_reward = DELIVERY_REWARD + distance_reward - delay_penalty - hop_penalty
                        rewards[idx] += delivery_reward
                        
                        # if self.time_slot < 3:
                        #     print(f"        DELIVERED! Packet reached destination GS{dst_gs}")
                    else:
                        # 转发奖励：基础转发奖励 + 距离奖励 - 小幅时延惩罚
                        forward_reward = FORWARD_REWARD + distance_reward - delay_penalty * 0.1
                        rewards[idx] += forward_reward
                        transit_packets.append(pkt)
                            
                    # 能耗惩罚
                    energy_cost = 0.01
                    node['energy'] -= energy_cost
                    cost_energy[idx] += energy_cost
                    rewards[idx] -= energy_cost * ENERGY_COST_WEIGHT
                    
                    # if self.time_slot < 3:
                    #     print(f"        Energy cost: {cost_energy[idx]:.3f}")
                else:
                    # 丢包：严重惩罚 + 时延惩罚
                    dropped_pkt = node['buffer'].pop(0)
                    cost_loss += 1
                    packet_age = self.time_slot - dropped_pkt['start_time']
                    drop_penalty = DROP_PENALTY - packet_age * DELAY_PENALTY_WEIGHT
                    rewards[idx] += drop_penalty
                    # if self.time_slot < 3:
                    #     print(f"        DROPPED! Target buffer full")
                    
            else:
                # 没有包可转发，小惩罚
                rewards[idx] += IDLE_PENALTY
                # if self.time_slot < 3 and idx >= self.num_satellites:
                #     print(f"        {node_type}: No packets to forward, penalty -0.05")

        # 计算系统级奖励
        system_reward = 0
        if delivered_packets:
            # 交付效率奖励
            delivery_efficiency = len(delivered_packets) / max(1, len(delivered_packets) + cost_loss)
            system_reward += delivery_efficiency * 1.0
            
            # 平均时延奖励（时延越短奖励越高）
            avg_delay = np.mean([self.time_slot - pkt['start_time'] for pkt in delivered_packets])
            delay_reward = max(0, (10 - avg_delay) * 0.1)  # 假设理想时延为10步以内
            system_reward += delay_reward
        
        # 将系统级奖励分配给所有agent
        rewards += system_reward / self.n_agents

        self.time_slot += 1

        # release queries scheduled for the new time slot
        self._release_queries()

        # update satellite positions for next slot if provided
        if self.sat_positions_per_slot is not None:
            idx = min(self.time_slot, len(self.sat_positions_per_slot) - 1)
            self.sat_positions = [np.array(p) for p in self.sat_positions_per_slot[idx]]

        done = self.time_slot >= self.max_time

        # 每隔3步补充新的包（模拟有流量），若提供 queries 则不再随机生成
        if not self.queries and self.time_slot % 3 == 0:
            for _ in range(self.random.randint(1, self.num_ground_stations)):
                gs_idx = self.random.randint(0, self.num_ground_stations - 1)
                gs = self.ground_stations[gs_idx]
                if len(gs['buffer']) < self.max_buffer:
                    pkt = self._create_packet()
                    gs['buffer'].append(pkt)

        obs = self.get_obs(neighbors)
        info = {
            'delivered_packets': len(delivered_packets),
            'packets_in_transit': len(transit_packets),
            'total_cost_loss': cost_loss,
            'delays': [self.time_slot - pkt['start_time'] for pkt in delivered_packets],
        }
        costs = {'energy': cost_energy, 'loss': cost_loss}
        
        return obs, rewards, done, costs, info




    def reset(self):
        self.satellites = self.initialize_satellites()
        self.ground_stations = self.initialize_ground_stations()
        self.time_slot = 0
        self.query_index = 0

        if self.sat_positions_per_slot is not None:
            self.sat_positions = [np.array(p) for p in self.sat_positions_per_slot[0]]

        self.neighbors = self._build_neighbors()
        
        # 初始化buffer：根据当前time_slot释放查询或随机生成
        self._release_queries()
        if not self.queries:
            for i in range(self.num_ground_stations):
                pkt = self._create_packet()
                self.ground_stations[i]['buffer'].append(pkt)

        return self.get_obs(self.neighbors)
