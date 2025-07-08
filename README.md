# CMADR
This is a implemention for CMADR algorithm.
Paper: [Dynamic Routing for Integrated Satellite-Terrestrial Networks: A Constrained Multi-Agent Reinforcement Learning Approach](https://ieeexplore.ieee.org/abstract/document/10436098/authors#authors)

## Usage

1. **Train**

   ```bash
   python train.py --config config.json
   ```

   The dataset name is given by `data_name` and the model will be saved to
   `model/<round>_<data_name>`. Training parameters such as the number of
   satellites, ground stations, queries and the maximum time slot are configured
   in `config.json`.

2. **Predict**

   ```bash
   python predict.py --config config.json
   ```

   The script loads the saved model and evaluates it on the dataset.

## 数据格式

项目的数据位于 `data/` 目录下，每个文件都是一个 JSON，主要字段说明如下：

- `num_satellites`：卫星数量。
- `num_ground`：地面站数量。
- `num_train_slots`、`num_predict_slots`：训练/预测阶段的时隙数。
- `num_train_queries`、`num_predict_queries`：训练/预测阶段的通信请求数量。
- `sat_positions_per_slot`：卫星在每个时隙的坐标，形状为 `[slot][sat][2]`。
- `gs_positions`：地面站的坐标列表。
- `train_queries`、`predict_queries`：请求列表，元素包含 `src`、`dst`、`time` 三个字段。

示例片段：

```json
{
    "num_satellites": 10,
    "num_ground": 5,
    "train_queries": [
        {"src": 1, "dst": 4, "time": 7},
        {"src": 1, "dst": 3, "time": 9}
    ]
}
```

## config 文件超参数

`config.json` 用于配置训练和预测的超参数，主要键值如下：

- `data_dir`：数据文件所在目录。
- `data_name`：数据文件名（不含后缀）。
- `model_root`：模型保存根目录。
- `round`：实验轮次，用于区分不同模型。
- `device`：使用的计算设备，如 `cpu` 或 `cuda`。
- `num_satellites`、`num_ground_stations`：卫星和地面站数量。
- `train_seed`、`predict_seed`：随机种子。
- `train.num_episodes`：训练轮数。
- `train.gamma`：奖励折扣因子。
- `train.cost_limits`：代价约束，例如能量和丢包限制。
- `train.train_queries`：生成训练请求的数量。
- `train.max_time`：训练阶段的时隙数。
- `predict.model_path`：预测时加载的模型路径。
- `predict.max_time`：预测阶段时隙数。
- `predict.predict_queries`：生成预测请求的数量。

## 项目结构

```
CMADR/
├── AC.py                # Actor-Critic 网络实现
├── ISTN_ENV.py          # 环境定义
├── LM.py                # 拉格朗日乘子和训练逻辑
├── MASys.py             # 多智能体系统封装
├── train.py             # 训练脚本
├── predict.py           # 预测脚本
├── data/                # 数据集目录
├── model/               # 训练得到的模型
├── config.json          # 默认配置文件
└── README.md            # 项目说明
```
