import json
from ISTN_ENV import ISTNEnv
from MASys import MultiAgentSystem
from LM import train_cmadr


def load_config(path: str) -> dict:
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


def main(config_path: str = 'config.json') -> None:
    cfg = load_config(config_path)
    with open(cfg['dataset_path'], 'r') as f:
        data = json.load(f)

    env = build_env(data)
    mac = MultiAgentSystem(
        n_agents=env.num_satellites + env.num_ground_stations,
        n_nodes=env.num_satellites + env.num_ground_stations,
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        hidden_dim=cfg.get('hidden_dim', 64),
        device=cfg.get('device', 'cpu'),
    )

    train_cmadr(
        env,
        mac,
        num_episodes=cfg.get('num_episodes', 100),
        gamma=cfg.get('gamma', 0.98),
        cost_limits=cfg.get('cost_limits', {'energy': 0.5, 'loss': 5}),
        device=cfg.get('device', 'cpu'),
    )

    mac.save(cfg.get('model_dir', 'model'))


if __name__ == '__main__':
    main()
