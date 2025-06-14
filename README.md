# CMDAR
This is a implemention for CMDAR algorithm.
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
