import json
import numpy as np
from ISTN_ENV import ISTNEnv
from MASys import MultiAgentSystem


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def load_data(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def build_env(data: dict) -> ISTNEnv:
    return ISTNEnv(
        num_satellites=len(data['sat_positions_per_slot'][0]),
        num_ground_stations=len(data['gs_positions']),
        max_time=len(data['sat_positions_per_slot']),
        sat_positions_per_slot=data['sat_positions_per_slot'],
        gs_positions=[tuple(p) for p in data['gs_positions']],
        queries=data['queries'],
    )


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


def predict(config_path: str = 'config.json'):
    cfg = load_config(config_path)
    data = load_data(cfg['dataset_path'])
    env = build_env(data)
    mac = MultiAgentSystem(
        n_agents=env.num_satellites + env.num_ground_stations,
        n_nodes=env.num_satellites + env.num_ground_stations,
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        hidden_dim=cfg.get('hidden_dim', 64),
        device=cfg.get('device', 'cpu'),
    )
    mac.load(cfg.get('model_dir', 'model'))

    loss_rate, energy, avg_delay = evaluate(env, mac)
    print(
        f"Prediction result: loss_rate={loss_rate:.3f}, energy={energy:.3f}, avg_delay={avg_delay:.3f}"
    )


if __name__ == "__main__":
    predict()
