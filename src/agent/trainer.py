import json
import os

from torch import optim
from torch.utils.tensorboard import SummaryWriter

from src.agent.agent import Agent
from src.agent.models.mlp import MLP
from src.agent.models.mm_policy import MMPolicyNetwork
from src.agent.ppo import PPOAgent, PPOParams
from src.agent.replay_buffer import ReplayBuffer
from src.lib import Dealer


class RLTrainer(Dealer):
    def __init__(self, agent: Agent, path="runs", rollout_length=2048):
        super().__init__()
        self.agent = agent
        self.rollout_length = rollout_length
        self.path = path

        # Ensure checkpoint directory exists
        if not os.path.exists(path):
            os.makedirs(path)

        files = os.listdir(path)
        # use only files that only have a number
        latest = [f for f in files if f.isnumeric()]
        latest = sorted(latest, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        self.newest_path = f'{path}/{len(latest)}'
        self.latest_path = f'{path}/{len(latest) - 1}' if len(latest) > 0 else None
        self.latest_path = f'{path}/{len(latest) - 1}' if len(latest) > 0 else None

        # TensorBoard writer
        self.optimizer = optim.Adam(
            self.agent.parameters(),
            betas=(0.9, 0.999),
            weight_decay=1e-5,
            amsgrad=True
        )
        # self.envs = [EnvInterface(router) for router in self.routers]

    def send(self, content, *args, **kwargs):
        msg = json.dumps(content).encode()
        response = super().send(msg)
        return response

    def train(self, num_episodes=1000, report=None, checkpoint=False):
        reward_history = []

        if not os.path.exists(self.newest_path):
            os.makedirs(self.newest_path)

        if report is None:
            def report(*args, **kwargs):
                pass

        if checkpoint:
            def save_checkpoint(i, reward, loss, values):
                if i % 10 == 0:
                    print(f'Episode {i}, Reward: {reward}, Loss: {loss}')
                    self.save(f"checkpoint_{i}")
                    if i > 0:
                        try:
                            os.remove(f'{self.newest_path}/checkpoint_{i - 10}.pth')
                        except FileNotFoundError:
                            pass

                for key, value in values.items():
                    writer.add_scalar(key, value, i)

        else:
            def save_checkpoint(i, reward, loss, values):
                pass

        writer = SummaryWriter(log_dir=self.newest_path)

        try:
            for episode in range(num_episodes):
                trajectories = self.collect_trajectories()
                # losses = self.agent.update(trajectories, self.optimizer)
                losses = {}
                episode_reward = sum(trajectories.rewards)
                reward_history.append(episode_reward)

                report({"reward": episode_reward})

                values = {key: value for key, value in losses.items()}
                values = {**values, "reward": episode_reward}

                save_checkpoint(episode, episode_reward, losses, values)

        except KeyboardInterrupt:
            pass

        writer.close()
        self.save()

        # return average 100 last rewards or
        n = min(100, len(reward_history))
        return sum(reward_history[-n:]) / n

    def collect_trajectories(self):
        trajectories = {
            env: ReplayBuffer()
            for env in self.routers
        }
        states = {
            env: env.reset()
            for env in self.routers
        }

        done = False

        for _ in range(self.rollout_length):
            for env in self.routers:
                action, log_prob, _ = self.agent(states[env])
                next_state, reward, done, trunc, _ = env.send(action)
                done = done or trunc
                trajectories[env].add(states[env], action, log_prob, reward, done)
                states[env] = next_state

            if done:
                break

        return trajectories

    def save(self, name="model"):
        self.agent.save_weights(f"{self.newest_path}/{name}.pth")

    def load(self):
        if not self.latest_path:
            raise FileNotFoundError('No runs found')
        file = f"model.pth"

        if not os.path.exists(f"{self.latest_path}/{file}"):
            files = os.listdir(self.latest_path)
            # get all files that start with checkpoint and end with .pth
            checkpoints = [f for f in files if f.startswith('checkpoint') and f.endswith('.pth')]
            if not checkpoints:
                raise FileNotFoundError('No weights found')
            latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]

            if not latest_checkpoint:
                raise FileNotFoundError('No weights found')
            file = latest_checkpoint
        self.agent.load_weights(f"{self.latest_path}/{file}")
        print(f'Loaded model from {self.latest_path}/{file}')

        return self.agent


if __name__ == "__main__":
    agent = PPOAgent(
        MMPolicyNetwork(
            in_features=3,
            in_depth=10,
            attention_heads=4,
            hidden_dims=(128, 128),
            out_dims=(4,)
        ),
        value_network=MLP(
            in_dim=3,
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
    trainer = RLTrainer(agent)
    trainer.use_mesh("pearl", "localhost", 8000)
    connect = {"type": "connect"}
    trainer.send(connect)

    trainer.train()
