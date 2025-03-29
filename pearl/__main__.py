import os
import multiprocessing
import time

import pearl.envs
import pearl.train
import pearl.mesh

def run_env(config=None):
    pearl.env.main(config)

def run_train(config=None, train_config=None, queue=None):
    result = pearl.train.main(config, train_config)
    if queue is not None:
        queue.put(result)

def run_mesh(config=None):
    pearl.mesh.main(config)


def main():
    n_envs = 2
    n_trainers = 2

    dotenv_config = {
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
    queue = multiprocessing.Queue()
    train_processes = [multiprocessing.Process(target=run_train, args=(dotenv_config, train_config, queue)) for _ in range(n_trainers)]
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
    while not queue.empty():
        print(queue.get())


if __name__ == "__main__":
    main()
