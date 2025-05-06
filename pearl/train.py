from pearl.config import MeshConfig, TrainConfig
from pearl.envs.multi.env_interface import EnvInterface

from pearl.agent.models.mlp import MLP
from pearl.agent.models.mm_policy import MMPolicyNetwork
from pearl.agent.ppo import PPOAgent, PPOParams
from pearl.agent.trainer import RLTrainer


def main(mesh_config: MeshConfig, train_config: TrainConfig) -> list:
    rollout_length, checkpoint_path, num_episodes, batch_size, learning_rate = train_config

    envs = EnvInterface()
    try:
        envs.use_mesh(mesh_config.name, mesh_config.host, mesh_config.port)
    except Exception as e:
        print(f"Failed to connect to mesh service: {e}")
        print(f"Are you sure the mesh service is running on {mesh_config.host}:{mesh_config.port} (try loading variables from .env)?")
        raise e

    envs.register_user()

    agent = PPOAgent(
        MMPolicyNetwork(
            in_features=8,
            in_depth=10,
            attention_heads=4,
            hidden_dims=(128, 128),
            out_dims=(int(1e2) for _ in range(4))
        ).cuda(),
        value_network=MLP(
            in_dim=48,
            hidden_units=(128, 128),
            out_dim=1
        ).cuda(),
        params=PPOParams(
            gamma=0.99,
            gae_lambda=0.95,
            eps_clip=0.2,
            entropy_coef=0.01,
        ),
        batch_size=batch_size,
        lr=learning_rate
    )

    trainer = RLTrainer(agent, envs, path=checkpoint_path, rollout_length=rollout_length)
    return trainer.train(num_episodes=num_episodes)
