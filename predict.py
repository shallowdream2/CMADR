import json
import argparse
import os
import numpy as np
from ISTN_ENV import ISTNEnv
from MASys import MultiAgentSystem


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def load_data(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def evaluate(env: ISTNEnv, mac: MultiAgentSystem):
    obs = env.reset()
    done = False
    total_loss = 0
    total_energy = 0.0
    delivered = 0
    total_delay = 0
    while not done:
        neighbors = env._build_neighbors()
        actions = mac.select_actions(obs, neighbors)
        obs, reward, done, costs, info = env.step(actions, neighbors)
        total_loss += costs['loss']
        total_energy += float(np.sum(costs['energy']))
        delivered += info['delivered_packets']
        total_delay += sum(info['delays'])
    loss_rate = total_loss / (delivered + total_loss) if delivered + total_loss > 0 else 0
    avg_delay = total_delay / delivered if delivered > 0 else 0
    return loss_rate, total_energy, avg_delay


def predict(config_path: str):
    cfg = load_config(config_path)
    data_dir = cfg.get('data_dir', 'data')
    data_name = cfg.get('data_name', 'dataset')
    data_path = os.path.join(data_dir, f"{data_name}.json")
    data = load_data(data_path)
    env = ISTNEnv(
        num_satellites=len(data['sat_positions_per_slot'][0]),
        num_ground_stations=len(data['gs_positions']),
        max_time=cfg.get('predict', {}).get('max_time', len(data['sat_positions_per_slot'])),
        sat_positions_per_slot=data['sat_positions_per_slot'],
        gs_positions=[tuple(p) for p in data['gs_positions']],
        queries=data['queries'],
    )
    mac = MultiAgentSystem(
        n_agents=env.num_satellites + env.num_ground_stations,
        n_nodes=env.num_satellites + env.num_ground_stations,
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        hidden_dim=64,
        device=cfg.get('device', 'cpu'),
    )

    model_dir = os.path.join(cfg.get('model_root', 'model'), f"{cfg.get('round',1)}_{data_name}")
    mac.load(model_dir)
    loss_rate, energy, avg_delay = evaluate(env, mac)
    print(
        f"Prediction result: loss_rate={loss_rate:.3f}, energy={energy:.3f}, avg_delay={avg_delay:.3f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.json', help='Path to config file')
    args = parser.parse_args()
    predict(args.config)
