import argparse
import os
import multiprocessing
import sys
import time

import numpy as np
import pandas as pd
import psutil

from rich.console import Console

import pearl.envs
import pearl.train
import pearl.mesh
from pearl.config import MeshConfig, TrainConfig


def run_mesh(mesh_config: MeshConfig):
    pearl.mesh.main(mesh_config)


def run_env(env_host, mesh_config: MeshConfig, n_envs: int, env_idx: int):
    p = psutil.Process(os.getpid())
    available_cpus = psutil.cpu_count(logical=True)
    core_id = env_idx % available_cpus
    p.cpu_affinity([core_id])

    pearl.env.main(env_host, mesh_config, n_envs, env_idx)


def run_train(mesh_config: MeshConfig, train_config: TrainConfig, queue=None):
    try:
        result, benchmark = pearl.train.main(mesh_config, train_config)
        if queue is not None:
            queue.put((result, benchmark))
    except KeyboardInterrupt:
        pass

def main(n_envs: int, n_trainers: int, mesh_config: MeshConfig, train_config: TrainConfig):
    mesh_process = multiprocessing.Process(target=run_mesh, args=(mesh_config,))
    mesh_process.start()
    time.sleep(2)

    # Start env processes first
    env_processes = [multiprocessing.Process(target=run_env, args=("localhost", mesh_config, n_envs, env_idx,))
                     for env_idx in range(n_envs)]
    for p in env_processes:
        p.start()
    time.sleep(2)

    # Start train processes
    queue = multiprocessing.Queue()
    train_processes = [multiprocessing.Process(target=run_train, args=(mesh_config, train_config, queue))
                       for _ in range(n_trainers)]
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
    trainer_idx = 0

    results = []

    while not queue.empty():
        f = queue.get()
        (wall_time, loss_history, updates), benchmark = f  # (time, reward)
        print(f"Trainer {trainer_idx} metrics:")
        trainer_idx += 1
        benchmark.report()

        results.append((np.array(wall_time), np.array(loss_history), np.array(updates)))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trainers", default=1, type=int)
    parser.add_argument("--n_envs", default=1, type=int)
    parser.add_argument("--k", default=1, type=int)

    MeshConfig.parser(parser)
    TrainConfig.parser(parser)

    args = parser.parse_args()

    config = {
        "n_trainers": args.n_trainers,
        "n_envs": args.n_envs,
        "k": args.k,
        "mesh": MeshConfig.parse_args(parser),
        "train": TrainConfig.parse_args(parser)
    }

    print(f"Running with config: {config}")

    out = pd.DataFrame()
    updates = pd.DataFrame()

    console = Console()
    try:
        for i in range(args.k):
            console.print(f"[green]Currently running n_envs={args.n_envs}, i={i}[/green]")
            results = main(args.n_envs, args.n_trainers, config["mesh"], config["train"])
            console.print(f"[green]Training finished successfully[/green]")
            for idx, result in enumerate(results):
                wall_time, loss, num_updates = result
                n = len(wall_time)
                df = pd.DataFrame({
                    "trainer_idx": [idx] * n,
                    "wall_time": wall_time[:, 0],
                    "loss": loss,
                    "n_envs": [args.n_envs] * n,
                    "iteration": [i] * n
                })
                num_updates = pd.DataFrame({
                    "trainer_idx": [idx],
                    "num_updates": num_updates,
                    "n_envs": [args.n_envs],
                    "iteration": [i]
                })
                out = pd.concat([out, df], ignore_index=True)
                updates = pd.concat([updates, num_updates], ignore_index=True)
    except KeyboardInterrupt:
        console.print(f"[yellow]Warning: Program interrupted prematurely by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


    out.to_csv("results.csv", index=False)
    updates.to_csv("updates.csv", index=False)
