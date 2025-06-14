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

        # 邻接表在每个time slot动态传入
        self.neighbors = None
        # 每个agent本地状态维度（自身+邻居+目的地距离）
        max_possible_neighbors = num_satellites + num_ground_stations - 1
        self.obs_dim = 3 + 3 * max_possible_neighbors + 1   # 末尾+1为“到目的地的距离”
        self.action_dim = num_satellites + num_ground_stations
        self.reset()

    def _build_neighbors(self):
        """Build neighbors based on current node positions."""
        neighbors = {i: set() for i in range(self.n_agents)}

        # satellite-satellite links
        for i in range(self.num_satellites):
            for j in range(self.num_satellites):
                if i == j:
                    continue
                dist = np.linalg.norm(np.array(self.sat_positions[i]) - np.array(self.sat_positions[j]))
                if dist <= self.conn_threshold:
                    neighbors[i].add(j)

        # satellite-ground links
        for gs in range(self.num_ground_stations):
            gs_pos = self.gs_positions[gs]
            for sat in range(self.num_satellites):
                dist = np.linalg.norm(np.array(self.sat_positions[sat]) - np.array(gs_pos))
                if dist <= self.conn_threshold:
                    neighbors[sat].add(self.num_satellites + gs)
                    neighbors[self.num_satellites + gs].add(sat)

        # convert sets to sorted lists
        return {k: sorted(list(v)) for k, v in neighbors.items()}

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
            current_neighbors = neighbors.get(i, [])
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
        
        # 调试：检查地面站buffer
        # if self.time_slot < 3:
        #     #print(f"    Ground stations buffer status:")
        #     for i in range(self.num_ground_stations):
        #         gs_agent_id = self.num_satellites + i  # 地面站的agent ID
        #         buffer_size = len(self.ground_stations[i]['buffer'])
        #         print(f"      GS {i} (Agent {gs_agent_id}): {buffer_size} packets")
        
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
                    rewards[idx] -= 0.1
                    continue

            # if self.time_slot < 3:
            #     buffer_size = len(node['buffer'])
            #     print(f"      Agent {idx}({node_type}): buffer_size={buffer_size}")

            if node['buffer']:
                pkt = node['buffer'][0]
                dst_gs = pkt['dst']
                
                # if self.time_slot < 3:
                #     print(f"        Found packet with dst={dst_gs}")
                
                # 检查动作有效性
                current_neighbors = neighbors.get(idx, [])
                if action >= len(current_neighbors):
                    # if self.time_slot < 3:
                    #     print(f"        Invalid action: {action} >= {len(current_neighbors)}")
                    rewards[idx] -= 0.1
                    continue
                    
                target = current_neighbors[action]
                
                # if self.time_slot < 3:
                #     print(f"        Trying to forward to target={target}")
                
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
                        # if self.time_slot < 3:
                        #     print(f"        Invalid target GS index: {tgt_gs_idx}")
                        rewards[idx] -= 0.1
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
                    
                    # 若目标是地面站且正好为目的地，则交付
                    if (target >= self.num_satellites) and ((target - self.num_satellites) == dst_gs):
                        delivered_packets.append(pkt)
                        tgt_node['buffer'].pop()  # 交付出队
                        rewards[idx] += 5.0  # 成功交付大奖励
                        # if self.time_slot < 3:
                        #     print(f"        DELIVERED! Packet reached destination GS{dst_gs}")
                    else:
                        # 基础转发奖励
                        rewards[idx] += 0.1
                        transit_packets.append(pkt)
                            
                    # 能耗
                    node['energy'] -= 0.01
                    cost_energy[idx] += 0.01
                    
                    # if self.time_slot < 3:
                    #     print(f"        Energy cost: {cost_energy[idx]:.3f}")
                else:
                    # 丢包
                    node['buffer'].pop(0)
                    cost_loss += 1
                    rewards[idx] -= 2.0  # 丢包惩罚
                    # if self.time_slot < 3:
                    #     print(f"        DROPPED! Target buffer full")
                    
            else:
                # 没有包可转发，小惩罚
                rewards[idx] -= 0.05
                # if self.time_slot < 3 and idx >= self.num_satellites:
                #     print(f"        {node_type}: No packets to forward, penalty -0.05")

        # 其余代码保持不变...
        # 添加全局奖励成分，但不要完全覆盖个体奖励
        if delivered_packets:
            global_bonus = len(delivered_packets) * 2.0
            rewards += global_bonus / self.n_agents  # 平均分配全局奖励

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
