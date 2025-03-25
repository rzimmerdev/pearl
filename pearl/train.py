from pearl.load_mesh import load_config
from pearl.envs.multi.env_interface import EnvInterface

from pearl.agent.models.mlp import MLP
from pearl.agent.models.mm_policy import MMPolicyNetwork
from pearl.agent.ppo import PPOAgent, PPOParams
from pearl.agent.trainer import RLTrainer


def main(config=None):
    host, mesh_name, mesh_host, mesh_port = load_config(config)

    envs = EnvInterface()
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
    )

    trainer = RLTrainer(agent, envs)
    trainer.train()


if __name__ == "__main__":
    main()
