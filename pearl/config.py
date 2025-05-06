from abc import ABC, abstractmethod
from argparse import ArgumentParser
from dataclasses import dataclass, astuple
from typing import Optional


@dataclass
class Config(ABC):
    @classmethod
    @abstractmethod
    def parser(cls, parser: ArgumentParser = None):
        pass

    @classmethod
    def parse_args(cls, parser: ArgumentParser = None, args: Optional[list[str]] = None):
        parser = parser or cls.parser(parser)
        parsed_args = parser.parse_args(args)

        # noinspection PyArgumentList
        return cls(**vars(parsed_args))

    def __iter__(self):
        # noinspection PyTypeChecker
        return iter(astuple(self))

@dataclass
class MeshConfig(Config):
    name: str = "pearl"
    host: str = "localhost"
    port: int = 5000

    @classmethod
    def parser(cls, parser: ArgumentParser = None) -> ArgumentParser:
        parser = parser or ArgumentParser(description="Mesh Configuration")
        parser.add_argument("--mesh_name", default=cls.name)
        parser.add_argument("--mesh_host", default=cls.host)
        parser.add_argument("--mesh_port", type=int, default=cls.port)
        return parser

    @classmethod
    def parse_args(cls, parser: ArgumentParser = None, args: Optional[list[str]] = None):
        parser = parser or cls.parser(parser)
        parsed_args = vars(parser.parse_args(args))

        # Map prefixed args to dataclass fields
        mapped_args = {
            'name': parsed_args['mesh_name'],
            'host': parsed_args['mesh_host'],
            'port': parsed_args['mesh_port'],
        }

        return cls(**mapped_args)

@dataclass
class TrainConfig(Config):
    rollout_length: int = 2048
    checkpoint_path: str = "runs/"
    num_episodes: int = 10
    batch_size: int = 64
    learning_rate: float = 1e-3

    @classmethod
    def parser(cls, parser: ArgumentParser = None) -> ArgumentParser:
        parser = parser or ArgumentParser(description="Train Configuration")
        parser.add_argument("--rollout_length", type=int, default=cls.rollout_length)
        parser.add_argument("--checkpoint_path", default=cls.checkpoint_path)
        parser.add_argument("--num_episodes", type=int, default=cls.num_episodes)
        parser.add_argument("--batch_size", type=int, default=cls.batch_size)
        parser.add_argument("--learning_rate", type=float, default=cls.learning_rate)

        return parser
