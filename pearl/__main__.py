import os
import multiprocessing
import time

import numpy as np
import pandas as pd
import psutil

from rich.console import Console

import pearl.envs
import pearl.train
import pearl.mesh

def run_env(n_envs, env_id, env_config):
    p = psutil.Process(os.getpid())
    available_cpus = psutil.cpu_count(logical=True)
    core_id = env_id % available_cpus
    p.cpu_affinity([core_id])

    # Your actual logic here
    pearl.env.main(n_envs, env_id, env_config)

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
    env_processes = [multiprocessing.Process(target=run_env, args=(n_envs, env_id, env_config,)) for env_id in range(n_envs)]
    for p in env_processes:
        p.start()
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

    for p in env_processes:
        p.join()
    mesh_process.join()

    while not queue.empty():
        wall_time, loss_history, updates = queue.get()  # (time, reward)

        return np.array(wall_time), np.array(loss_history), np.array(updates)


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

    k = 10

    out = pd.DataFrame()
    updates = pd.DataFrame()
    # ["wall_time", "loss", "num_updates", "n_env", "iteration"]
    # set the columns to the above

    console = Console()
    try:
        for n_env in [1, 2, 4, 8, 16, 32]:
            for i in range(k):
                console.print(f"[green]Running with n_env={n_env}, i={i}[/green]")
                wall_time, loss, num_updates = main(n_env, n_trainers, env_config, train_config)
                n = len(wall_time)
                df = pd.DataFrame({
                    "wall_time": wall_time[:, 0],
                    "loss": loss,
                    "n_env": [n_env] * n,
                    "iteration": [i] * n
                })
                num_updates = pd.DataFrame({
                    "num_updates": [num_updates],
                    "n_env": [n_env],
                    "iteration": [i]
                })
                out = pd.concat([out, df], ignore_index=True)
                updates = pd.concat([updates, num_updates], ignore_index=True)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")

    # save to csv
    out.to_csv("results.csv", index=False)
    updates.to_csv("updates.csv", index=False)
