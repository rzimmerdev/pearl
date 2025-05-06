import inspect
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from dataclasses import dataclass, astuple


@dataclass
class Config(ABC):
    @classmethod
    @abstractmethod
    def parser(cls, parser: ArgumentParser = None):
        pass

    @classmethod
    def parse_args(cls, parser: ArgumentParser = None, args = None):
        parser = parser or cls.parser(parser)
        parsed_args = parser.parse_args(args)

        # noinspection PyArgumentList
        return cls.from_dict(dict(**vars(parsed_args)))

    def __iter__(self):
        # noinspection PyTypeChecker
        return iter(astuple(self))

    @classmethod
    def from_dict(cls, args):
        # noinspection PyArgumentList
        return cls(**{
            k: v for k, v in args.items()
            if k in inspect.signature(cls).parameters
        })

@dataclass
class MeshConfig(Config):
    name: str = "pearl"
    host: str = "localhost"
    port: int = 5000

    @classmethod
    def parser(cls, parser: ArgumentParser = None) -> ArgumentParser:
        parser = parser or ArgumentParser(description="Mesh Configuration")
        parser.add_argument("--name", default=cls.name)
        parser.add_argument("--host", default=cls.host)
        parser.add_argument("--port", type=int, default=cls.port)
        return parser


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
