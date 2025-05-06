from dataclasses import dataclass


@dataclass
class MeshConfig:
    name: str = "localhost"
    host: str = "localhost"
    port: int = 5000


@dataclass
class TrainConfig:
    rollout_length: int = 2048
    checkpoint_path: str = "runs/"
    num_episodes: int = 10
    batch_size: int = 64
    learning_rate: float = 1e-3