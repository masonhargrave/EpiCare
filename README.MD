# EpiCare: Episodes of Care Library

Welcome to EpiCare, a benchmarking library for reinforcement learning in healthcare applications. This ReadMe provides a guide to installing, configuring, and utilizing EpiCare.

## Installation

To get started with EpiCare, please follow the installation steps below:

### Before Installing 

Create a new conda environment with python version 3.9+ using
```bash
conda create --name EpiCare python==python3.10
```

Ensure you have PyTorch installed on your system. Visit [PyTorch's official installation guide](https://pytorch.org/get-started/locally/) for detailed instructions.

### Installing EpiCare

Clone the EpiCare repository to your local machine and navigate to the root directory. Then, install EpiCare along with its dependencies by running:

```bash
pip install -e .
```

EpiCare depends on the following Python packages:

- numpy
- h5py
- gym=0.23
- pyrallis
- tqdm
- wandb
- scipy
- pandas

These dependencies will be automatically installed during the EpiCare installation process.

## The EpiCare Environment

The EpiCare environmet is an Gym environment which simulates longitudinal patient care. Full details about the environment can be found in our paper. To load the environment run:

```python
from epicare.envs import EpiCare

env = EpiCare()
```

The environment is highly customizable, take a look at the paper for inspiration on what might be worth configuring for your own projects. See `figures/figures.ipynb` for some examples of running and evaluating custom policies on this environment.

If you would like to use `gym` to test any methods on this environment, simply run

```python
import epicare.envs # This registers the environment automatically
import gym

gym.make("EpiCare-v0")
```

## Included Offline RL Implementations

In addition to the EpiCare environment, we have included single-file implementations for five offline RL methods adapted to work for discrete control environments

- AWAC
- EDAC
- TD3+BC
- CQL
- IQL
- DQN
- DRN (Deep Reward Network)
- BC

## Reproducing our Results

### Generating Training Data

To generate the necessary training data, navigate to the `data` directory and execute:

```bash
python data_gen.py
```

This script prepares the data required for training the included offline RL models as well as any other models you may want to test. This data conforms by-and-large to the D4RL standards and can be loaded for other RL pipelines using our data loader found in `epicare.utils.load_custom_datset`.

### Hyperparameter Optimization

EpiCare integrates with Weights & Biases (W&B) for hyperparameter optimization. Follow these steps to perform a hyperparameter sweep:

1. **W&B Setup**: If you haven't already, sign up or log in to your W&B account at [wandb.ai](https://wandb.ai). Use the `wandb login` command to authenticate your CLI.
2. **Creating a Sweep**:

   - Choose a configuration file corresponding to the RL model of your choice from the `algorithms/sweep_configs/hp_sweeps` directory.
   - Create a hyperparameter sweep project by running:
     ```bash
     wandb sweep --project EpiCare [CONFIG FILE]
     ```
   - This command initializes the project on W&B and outputs instructions for launching agent instances.

3. **Running the Sweep**:

   - Execute the provided command in one or multiple instances (across different machines if necessary) to start the hyperparameter optimization process.
   - Monitor the progress and results on the W&B web dashboard.

4. **Selecting Hyperparameters**:
   - You may choose the optimal hyperparameters manually from the dashboard or utilize the defaults found in our algorithms, which are based on comprehensive testing as detailed in the paper corresponding to this repository.

### Training Models

With the hyperparameters set, you can proceed to train the models. EpiCare's framework allows you to queue the training of multiple replicates across multiple environment seed using a single command by way of another sweep process.

- For each configuration in `algorithms/sweep_configs/all_data_sweeps`, initiate a sweep with:
  ```bash
  wandb sweep --project EpiCare [CONFIG FILE]
  ```
- Follow the resulting instructions to launch agents to train your selected model.
- Model checkpoints are saved periodically throughout training runs in the `algorithms/checkpoints` directory

To customize hyperparameters beyond the defaults, modify the code directly or update the configuration files accordingly.

### Data-Restricted Training

EpiCare offers functionality for training under data-restricted scenarios. This feature focuses on the first environment seed and varies the number of training episodes (though in principle the experiment could be run for any environment seed of your choosing). Configuration files for the three best-performing models are located in `algorithms/sweep_configs/data_restriction_sweeps`. Checkpoints for these runs are only saved at the very end in the `algorithms/checkpoints` directory.

### Evaluating Models After Training

To evaluate the final checkpoint of each run, simply run `python algorithms/awac.py evaluate`, and the evaluation results will be output to `results/awac_results.csv`.

These instructions work for all other algorithms provided simply by swapping out the algorithm name in the commands (i.e. subbing `edac.py` instead of `awac.py`).

### Final Analysis

In the `figures` directory we have included a jupyter notebook with examples for how we analyzed our results and produced most of the figures found in our paper. After running all full-data and data restriction trials as described above, this notebook allows you to reproduce the main results of the paper.

The python packages required to run `figures.ipynb` are included in `requirements.txt`. To install, run `pip install -f requirements.txt` while in the `figures` directory.
