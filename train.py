import json
import argparse
import os
from ISTN_ENV import ISTNEnv
from MASys import MultiAgentSystem
from LM import train_cmadr


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def build_env_from_data(data: dict, max_time: int | None = None) -> ISTNEnv:
    return ISTNEnv(
        num_satellites=len(data['sat_positions_per_slot'][0]),
        num_ground_stations=len(data['gs_positions']),
        max_time=max_time or len(data['sat_positions_per_slot']),
        sat_positions_per_slot=data['sat_positions_per_slot'],
        gs_positions=[tuple(p) for p in data['gs_positions']],
        queries=data['queries'],
    )


def main(config_path: str):
    cfg = load_config(config_path)
    data_dir = cfg.get('data_dir', 'data')
    data_name = cfg.get('data_name', 'dataset')
    data_path = os.path.join(data_dir, f"{data_name}.json")

    train_cfg = cfg.get('train', {})
    if not os.path.exists(data_path):
        from data_generator import generate_dataset
        data = generate_dataset(
            train_cfg.get('num_satellites', 5),
            train_cfg.get('num_ground_stations', 5),
            train_cfg.get('num_queries', 10),
            train_cfg.get('max_time', 50),
        )
        os.makedirs(data_dir, exist_ok=True)
        with open(data_path, 'w') as f:
            json.dump(data, f, indent=2)
    else:
        with open(data_path, 'r') as f:
            data = json.load(f)

    env = build_env_from_data(data, train_cfg.get('max_time'))
    mac = MultiAgentSystem(
        n_agents=env.num_satellites + env.num_ground_stations,
        n_nodes=env.num_satellites + env.num_ground_stations,
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        hidden_dim=64,
        device=cfg.get('device', 'cpu'),
    )

    model_dir = os.path.join(cfg.get('model_root', 'model'), f"{cfg.get('round',1)}_{data_name}")
    train_cmadr(
        env,
        mac,
        num_episodes=train_cfg.get('num_episodes', 10),
        gamma=train_cfg.get('gamma', 0.98),
        cost_limits=train_cfg.get('cost_limits', {'energy': 0.5, 'loss': 5}),
        device=cfg.get('device', 'cpu'),
    )

    mac.save(model_dir)
    print(f"Model saved to {model_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.json', help='Path to config file')
    args = parser.parse_args()
    main(args.config)
