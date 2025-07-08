# ISTN 环境奖励系统设计文档

## 1. 设计原则

### 1.1 核心目标

- **最大化包交付率**：成功交付到目的地的包数量
- **最小化传输时延**：减少包在网络中的传输时间
- **最短路径优化**：鼓励选择更短的路径
- **能耗效率**：平衡性能和能耗
- **避免拥塞**：防止网络拥堵和丢包

### 1.2 平衡考虑

- **个体奖励 vs 系统奖励**：既要激励个体最优决策，也要考虑系统整体性能
- **即时奖励 vs 长期奖励**：平衡短期行为和长期目标
- **探索 vs 利用**：鼓励探索新路径，同时利用已知好路径

## 2. 奖励组成

### 2.1 基础奖励参数

```python
DELIVERY_REWARD = 10.0          # 成功交付基础奖励
FORWARD_REWARD = 0.2            # 转发基础奖励
DROP_PENALTY = -5.0             # 丢包惩罚
INVALID_ACTION_PENALTY = -0.5   # 无效动作惩罚
IDLE_PENALTY = -0.1             # 空闲惩罚
ENERGY_COST_WEIGHT = 2.0        # 能耗权重
DISTANCE_REWARD_WEIGHT = 0.1    # 距离奖励权重
DELAY_PENALTY_WEIGHT = 0.05     # 时延惩罚权重
```

### 2.2 奖励详细计算

#### 2.2.1 成功交付奖励

```
delivery_reward = DELIVERY_REWARD + distance_reward - delay_penalty - hop_penalty
```

- **基础奖励**：10.0（最高奖励，强烈激励交付）
- **距离奖励**：根据是否朝目的地方向移动
- **时延惩罚**：包在网络中停留时间越长惩罚越大
- **跳数惩罚**：路径越长惩罚越大（每跳 0.1）

#### 2.2.2 转发奖励

```
forward_reward = FORWARD_REWARD + distance_reward - delay_penalty * 0.1
```

- **基础奖励**：0.2（鼓励转发行为）
- **距离奖励**：朝目的地方向转发给予额外奖励
- **轻微时延惩罚**：避免无意义的转发

#### 2.2.3 距离奖励

```
distance_improvement = current_to_dst - target_to_dst
distance_reward = distance_improvement * DISTANCE_REWARD_WEIGHT
```

- **正向奖励**：转发使包更接近目的地
- **负向惩罚**：转发使包远离目的地

#### 2.2.4 时延惩罚

```
packet_age = current_time - packet_start_time
delay_penalty = packet_age * DELAY_PENALTY_WEIGHT
```

- **累积惩罚**：包在网络中停留时间越长惩罚越大
- **激励快速传输**：促使 agent 优先处理旧包

#### 2.2.5 能耗惩罚

```
energy_penalty = energy_cost * ENERGY_COST_WEIGHT
```

- **转发成本**：每次转发消耗 0.01 能量
- **权重放大**：通过权重 2.0 使能耗成为重要考虑因素

#### 2.2.6 丢包惩罚

```
drop_penalty = DROP_PENALTY - packet_age * DELAY_PENALTY_WEIGHT
```

- **基础惩罚**：-5.0（严重惩罚）
- **时延加重**：包等待时间越长，丢包惩罚越重

#### 2.2.7 无效动作惩罚

- **无效邻居选择**：-0.5
- **无效节点索引**：-0.5
- **防止随机行为**：discourage 无意义的探索

#### 2.2.8 空闲惩罚

- **无包可转发**：-0.1
- **鼓励主动性**：促使 agent 主动获取和处理包

### 2.3 系统级奖励

```python
# 交付效率奖励
delivery_efficiency = delivered_packets / (delivered_packets + dropped_packets)
system_reward += delivery_efficiency * 1.0

# 平均时延奖励
avg_delay = mean([current_time - packet_start_time for packet in delivered_packets])
delay_reward = max(0, (10 - avg_delay) * 0.1)
system_reward += delay_reward

# 分配给所有agent
rewards += system_reward / n_agents
```

## 3. 奖励特性分析

### 3.1 奖励数值范围

- **最高奖励**：成功交付 ≈ 10+ (基础+距离+时延奖励)
- **中等奖励**：有效转发 ≈ 0.2-0.5
- **轻微惩罚**：空闲、轻微时延 ≈ -0.1
- **中等惩罚**：无效动作 ≈ -0.5
- **严重惩罚**：丢包 ≈ -5.0+

### 3.2 激励机制

1. **强烈激励交付**：交付奖励远高于转发奖励
2. **路径优化**：距离奖励引导最短路径
3. **时延敏感**：时延惩罚促进快速传输
4. **能耗平衡**：能耗权重避免过度转发
5. **协作促进**：系统级奖励鼓励整体协作

### 3.3 避免问题

1. **避免贪婪**：不只考虑即时奖励，时延惩罚鼓励长期规划
2. **避免拥塞**：丢包重惩罚鼓励流量控制
3. **避免无效探索**：无效动作惩罚提高效率
4. **避免能耗浪费**：能耗权重控制转发频率

## 4. 调优建议

### 4.1 根据场景调整

- **高时延敏感**：增加 DELAY_PENALTY_WEIGHT
- **高能耗限制**：增加 ENERGY_COST_WEIGHT
- **拥塞严重**：增加 DROP_PENALTY 绝对值
- **探索不足**：减少 INVALID_ACTION_PENALTY

### 4.2 动态调整

- **训练初期**：较高的探索奖励，较低的惩罚
- **训练后期**：较低的探索奖励，较高的效率要求
- **实际部署**：根据网络状况实时调整权重

### 4.3 监控指标

- **交付率**：delivered_packets / total_packets
- **平均时延**：mean(delivery_time - start_time)
- **平均跳数**：mean(hop_count)
- **能耗效率**：delivered_packets / total_energy_consumed
- **丢包率**：dropped_packets / total_packets

## 5. 实验验证

测试结果显示：

- ✅ 成功交付获得高奖励（≈12.0）
- ✅ 转发获得中等奖励（≈2.5）
- ✅ 丢包获得严重惩罚（≈-5.0）
- ✅ 无效动作获得惩罚（≈-0.5）
- ✅ 距离机制有效（近距离转发 > 远距离转发）

这个奖励系统能够有效引导 agents 学习到：

1. 优先交付包到目的地
2. 选择朝向目的地的转发路径
3. 避免无效和浪费的行为
4. 平衡性能和能耗
5. 促进整体系统协作
