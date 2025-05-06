# Parallel Environments for Asynchronous RL

## Get Started

Requirements 
- [Python 3.10](https://www.python.org/downloads/release/python-3100/) or above.
- [Pip](https://pip.pypa.io/en/stable/installation/) or some other package manager.
- [ZeroMQ](https://zeromq.org/) for the messaging queue system.

Clone the repository locally

```bash
git clone https://github.com/rzimmerdev/tcc
```

Or download the latest release from [Release](google.com) and unpack it.

```bash
tar -xzvf release.tar.gz
```

Enter the repository

```bash
cd tcc
```
To run the `pearl` module, you will need to install the required dependencies. 
If using `pip`, run

```bash
pip install -r requirements.txt
```


## Using the Pearl Module

To get a sense of how to use the Pearl framework, call the main module with the `-h` or `--help` arguments.

```bash
python -m pearl --help
```
```
usage: __main__.py [-h] 
                   [--n_trainers N_TRAINERS] 
                   [--n_envs N_ENVS] [--k K] 
                   [--name NAME] [--host HOST] [--port PORT] 
                   [--rollout_length ROLLOUT_LENGTH] 
                   [--checkpoint_path CHECKPOINT_PATH] 
                   [--num_episodes NUM_EPISODES]
                   [--batch_size BATCH_SIZE] 
                   [--learning_rate LEARNING_RATE]

options:
  -h, --help            show this help message 
                        and exit
  --n_trainers N_TRAINERS
  --n_envs N_ENVS
  --k K
  --name NAME
  --host HOST
  --port PORT
  --rollout_length ROLLOUT_LENGTH
  --checkpoint_path CHECKPOINT_PATH
  --num_episodes NUM_EPISODES
  --batch_size BATCH_SIZE
  --learning_rate LEARNING_RATE

```

## Additional Details

To start the cluster of parallel environments for async Reinforcement Learning (PEARL cluster), 
call the main module, and pass any number of arguments to configure the execution:

```bash
python -m pearl --k 1 --n_envs 32 --num_episodes 10 --learning_rate 1e-3
```

```
Running with config: {'n_trainers': 1, 'n_envs': 32, 'k': 1, 'mesh': MeshConfig(name='pearl', host='localhost', port=5000), 'train': TrainConfig(rollout_length=2048, checkpoint_path='runs/', num_episodes=10, batch_size=64, learning_rate=0.001)}
Currently running n_envs=32, i=0
...
```

You will then see a sequence of events in the terminal, as the cluster is started and connects the agents to the environments. 
The cluster is composed of the following services:
1. The mesh server for service discovery within the cluster.
2. Parallel individual environments, running in separate process or machine.
3. Learner processes, which are responsible for training the agents and can connect to multiple individual environments.

After connection is established, the trainers will send messages until step-timeout, where they will retry to send the message
or wait for the next step, or until training is finished. 

Training finishes when all trainers completed the desired number of episodes.
Then, a simple benchmark will be printed to the console:

```
trainer.act     | avg: 0.003481s | calls: 527 | total: 1.834559s
trainer.step    | avg: 0.005713s | calls: 526 | total: 3.005092s
trainer.reset   | avg: 0.000005s | calls: 526 | total: 0.002862s
```

showing how much time each trainer spent on average, as well as how many calls were made for specific
parts of the code. Two files containing in-depth result metrics will be saved to `results.csv` for 
RL related metrics, and to `updates.csv` related to number of total updates performed.

These files are indexed by number of environments and number of trainers used,
to facilitate generating interesting graphics. 
The code used to generate such graphics is available under the `results/` directory.

## Thesis

Authored by Rafael Zimmer, 2025.
View the complete document here, or simply access the [file's latest commit](paper/main/out/main.pdf).