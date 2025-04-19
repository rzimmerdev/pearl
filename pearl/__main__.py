import os
import multiprocessing
import time
from typing import List

import numpy as np

import pearl.envs
import pearl.train
import pearl.mesh

def run_env(max_envs: int, config=None):
    pearl.env.main(max_envs, config)

def run_train(config=None, train_config=None, queue=None):
    result = pearl.train.main(config, train_config)
    if queue is not None:
        queue.put(result)

def run_mesh(config=None):
    pearl.mesh.main(config)


def main(n_envs, n_trainers, env_config, train_config):
    mesh_process = multiprocessing.Process(target=run_mesh, args=(env_config,))
    mesh_process.start()

    time.sleep(2)

    # Start env processes first
    env_processes = [multiprocessing.Process(target=run_env, args=(n_envs, env_config,)) for _ in range(n_envs)]
    for p in env_processes:
        p.start()
        time.sleep(.5)

    time.sleep(2)

    # Start train processes
    queue = multiprocessing.Queue()
    train_processes = [multiprocessing.Process(target=run_train, args=(env_config, train_config, queue)) for _ in range(n_trainers)]
    for p in train_processes:
        p.start()

    try:
        for p in train_processes:
            p.join()
    except KeyboardInterrupt:
        pass

    for p in env_processes:
        p.terminate()
    mesh_process.terminate()

    # Optionally, join to ensure cleanup
    for p in env_processes:
        p.join()
    mesh_process.join()

    print("Training ended.")
    print("Results:")
    wall_time = []
    loss = []
    total_updates = []
    while not queue.empty():
        reward_history, loss_history, updates = queue.get()  # (time, reward)
        wall_time.append(reward_history[-1][0])
        loss.append(loss_history[-1])
        total_updates.append(updates)

    wall_time = np.array(wall_time)
    return wall_time.mean(), np.array(loss).mean(), np.array(total_updates).mean()


if __name__ == "__main__":
    n_trainers = 1

    env_config = {
        "HOST": os.getenv("HOST", "localhost"),
        "MESH_NAME": os.getenv("MESH_NAME", "pearl"),
        "MESH_HOST": os.getenv("MESH_HOST", "localhost"),
        "MESH_PORT": os.getenv("MESH_PORT", 5000),
    }

    train_config = {
        "ROLLOUT_LENGTH": os.getenv("ROLLOUT_LENGTH", 2048),
        "PATH": os.getenv("PATH", "runs"),
        "NUM_EPISODES": os.getenv("NUM_EPISODES", 10),
        "BATCH_SIZE": os.getenv("BATCH_SIZE", 64),
    }

    for n_env in [1, 2, 4, 8, 16, 32]:
        print(f"Running with {n_env} envs and {n_trainers} trainers.")
        wall_time, loss, num_updates = main(n_env, n_trainers, env_config, train_config)
        print(f"Wall time: {wall_time}, Loss: {loss}, Num Updates: {num_updates}")