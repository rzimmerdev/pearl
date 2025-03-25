TCC

Author: Rafael Zimmer

To start the cluster of parallel environments for async Reinforcement Learning (PEARL cluster), run:

```bash
$ python3 -m pearl
```

You will then see a sequence of events in the terminal, as the cluster is started. The cluster is composed of the following services:
1. The mesh server for service discovery within the cluster.
2. Parallel individual environments, running in separate process or machine.
3. Learner processes, which are responsible for training the agents and can connect to multiple individual environments.

Learner processes can be l
