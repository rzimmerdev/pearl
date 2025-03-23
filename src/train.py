import os

from src.agent.models.mlp import MLP
from src.agent.models.mm_policy import MMPolicyNetwork
from src.agent.ppo import PPOAgent, PPOParams
from src.agent.trainer import RLTrainer
from src.env.multi.env_interface import EnvInterface


def main():
    envs = EnvInterface()

    mesh_name = os.getenv("MESH_NAME", "pearl")
    mesh_host = os.getenv("MESH_HOST", "localhost")
    mesh_port = int(os.getenv("MESH_PORT", "5000"))

    try:
        envs.use_mesh(mesh_name, mesh_host, mesh_port)
    except Exception as e:
        print(f"Failed to connect to mesh service: {e}")
        print(f"Are you sure the mesh service is running on {mesh_host}:{mesh_port} (try loading variables from .env)?")
        return

    envs.register_user()

    agent = PPOAgent(
        MMPolicyNetwork(
            in_features=8,
            in_depth=10,
            attention_heads=4,
            hidden_dims=(128, 128),
            out_dims=(int(1e2) for _ in range(4))
        ),
        value_network=MLP(
            in_dim=48,
            hidden_units=(128, 128),
            out_dim=1
        ),
        params=PPOParams(
            gamma=0.99,
            gae_lambda=0.95,
            eps_clip=0.2,
            entropy_coef=0.01,
        ),
    )

    trainer = RLTrainer(agent, envs)
    trainer.train()


if __name__ == "__main__":
    main()
