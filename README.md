[**Prerequisites**](#prerequisites)
| [**Install guide**](#install-guide)
| [**Experiments**](#experiments)
| [**Contact**](#contact) 

# The impact of time: lifetime as governing reward in RL
#### Author(s):
Arnau Quindós
#### Manuscript:
_[WIP]_
#### Abstract 
In this study, we investigate the idea of using the agent's lifetime as reward in reinforcement learning, motivated by the fact that good reward design usually requires in-depth task-specific knowledge and the universal nature of time. We divide the research into two different methodologies: firstly, we examine how time could improve reward shaping, with results indicating that time can serve as a reward shaper and suggesting a bigger potential for using time as the sole reward. In our second approach, we train agents in the absence of explicit goal-oriented rewards, under the assumption that termination probabilities inherently encode the goals of the problem, an idea observed in many real-world settings. Results show that time-based agents can come close to or even surpass agents trained in the traditional setting both in end performance and learning speed. Our findings highlight the significance and potential of an agent's lifetime in reward design but with significant challenges that need to be overcome. Further research is needed to validate our methodologies in diverse and more complex environments and to explore ways of generalising this idea to a broader spectrum of reinforcement learning domains.

## Prerequisites
* Anaconda python > 3.10

## Install guide

In order to ensure compatibility, create a conda environment.
```sh
conda create --name minigrid python=3.10
conda activate minigrid
```

Install the following python packages.
```sh
pip install torch gymnasium==0.28.1 minigrid==2.2.1 matplotlib==3.7.1 wandb==0.15.4 imageio==2.31.1
```
## Experiments

To train a single agent, use the following script:

```sh
python train.py [OPTIONS]
```

Some of the relevant options are:

- `--env-id`: ID of the environment. Default is `MiniGrid-Empty-6x6-v0`.
- `--seed`: Seed of the experiment. Default is `1`.
- `--verbose`: Print metrics and training logs. Default is `True`.
- `--wandb`: Use wandb to log metrics. Default is `True`.
- `--wandb-project`: Wandb project name. Default is `experiments-test`.
- `--total-timesteps`: Total timesteps of the experiments. Default is `1000000`.
- `--num-envs`: The number of parallel game environments. Default is `32`.
- `--num-steps`: The number of steps to run in each environment per policy rollout. Default is `256`.

To analyse the behaviour of a fully-trained agent, use the `exploitation.py` script, use the following command:

```bash
python exploitation.py [OPTION 1] [OPTION 2] [...]
```

Some relevant options:

- `--env-id`: ID of the environment. Default is `EnergyBoxes`.
- `--max-timesteps`: Maximum timesteps for the exploitation. Default is `256`.
- `--capture-gif`: Capture the agent's performance as a GIF. Default is `False`.
- `--agent-name`: Name of the agent. Default is `test`.
- `--render-mode`: Mode for rendering the environment for visualizing agent behaviour. Default is `human`.


### Replicating Experiments from the Manuscript

To replicate the experiments described in the manuscript, you can use the provided script on an HPC cluster. Follow the steps below:

1. Ensure you have cloned the repository and navigated to the project directory.

2. Use the `run-experiments.sh` script with the described options:

```bash
./run-experiments.sh [COMPUTE_TYPE] [CSV_FILE] [SEED_RANGE]
```

- `[COMPUTE_TYPE]`: Choose between `cpu` or `gpu`.
- `[CSV_FILE]`: CSV file containing experiment configurations.
- `[SEED_RANGE]`: Range of seeds for reproducibility. Specify as `min_seed-max_seed` (e.g., `1-5`).

For example, to replicate the experiments using the CPU, the `experiments-1.csv` file, and seeds ranging from 1 to 5, use:

```bash
./run-experiments.sh cpu experiments-1.csv 1-5
```


## Contact
* arnau.quindos.22@ucl.ac.uk
