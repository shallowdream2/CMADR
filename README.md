# CMDAR
This is a implemention for CMDAR algorithm.
Paper: [Dynamic Routing for Integrated Satellite-Terrestrial Networks: A Constrained Multi-Agent Reinforcement Learning Approach](https://ieeexplore.ieee.org/abstract/document/10436098/authors#authors)

## Usage

1. Adjust the hyperparameters in `config.json`. The file contains all settings for training and inference, including the path to the dataset and the directory used to store trained models.
2. Run training
   ```bash
   python train.py
   ```
   Trained weights will be saved under the directory specified by `model_dir` in the config file.
3. Run prediction
   ```bash
   python predict.py
   ```
   The script loads the saved model and reports loss rate, energy consumption and average delay.
