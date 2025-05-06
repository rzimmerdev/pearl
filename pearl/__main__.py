import argparse
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
from pearl.config import MeshConfig, TrainConfig


def run_env(env_host, mesh_config: MeshConfig, n_envs: int, env_idx: int):
    p = psutil.Process(os.getpid())
    available_cpus = psutil.cpu_count(logical=True)
    core_id = env_idx % available_cpus
    p.cpu_affinity([core_id])

    pearl.env.main(env_host, mesh_config, n_envs, env_idx)


def run_train(mesh_config: MeshConfig, train_config: TrainConfig, queue=None):
    result = pearl.train.main(mesh_config, train_config)
    if queue is not None:
        queue.put(result)


def run_mesh(mesh_config: MeshConfig):
    pearl.mesh.main(mesh_config)


def main(n_envs: int, n_trainers: int, mesh_config: MeshConfig, train_config: TrainConfig):
    mesh_process = multiprocessing.Process(target=run_mesh, args=(mesh_config,))
    mesh_process.start()
    time.sleep(2)

    # Start env processes first
    env_processes = [multiprocessing.Process(target=run_env, args=(n_envs, env_id, mesh_config,)) for env_id in
                     range(n_envs)]
    for p in env_processes:
        p.start()
    time.sleep(2)

    # Start train processes
    queue = multiprocessing.Queue()
    train_processes = [multiprocessing.Process(target=run_train, args=(mesh_config, train_config, queue)) for _ in
                       range(n_trainers)]
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--")

    k = 10

    out = pd.DataFrame()
    updates = pd.DataFrame()

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
