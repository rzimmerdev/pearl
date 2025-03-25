import os
import multiprocessing
import time

import pearl.envs
import pearl.train
import pearl.mesh

def run_env(config=None):
    pearl.env.main(config)

def run_train(config=None):
    pearl.train.main(config)

def run_mesh(config=None):
    pearl.mesh.main(config)

if __name__ == "__main__":
    n_envs = 2
    n_trainers = 2

    dotenv_config = {
        "HOST": os.getenv("HOST", "localhost"),
        "MESH_NAME": os.getenv("MESH_NAME", "pearl"),
        "MESH_HOST": os.getenv("MESH_HOST", "localhost"),
        "MESH_PORT": os.getenv("MESH_PORT", 5000),
    }

    mesh_process = multiprocessing.Process(target=run_mesh, args=(dotenv_config,))
    mesh_process.start()

    time.sleep(2)

    # Start env processes first
    env_processes = [multiprocessing.Process(target=run_env, args=(dotenv_config,)) for _ in range(n_envs)]
    for p in env_processes:
        p.start()
        time.sleep(.5)

    time.sleep(2)

    # Start train processes
    train_processes = [multiprocessing.Process(target=run_train) for _ in range(n_trainers)]
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
