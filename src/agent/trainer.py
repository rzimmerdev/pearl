import json
import os
from typing import Dict

import torch
from torch.utils.tensorboard import SummaryWriter

from src.agent.agent import Agent
from src.agent.replay_buffer import ReplayBuffer
from src.env.multi.env_interface import EnvInterface
from src.lib.benchmark import Benchmark


class RLTrainer:
    def __init__(self, agent: Agent, envs: EnvInterface, path="runs", rollout_length=2048):
        super().__init__()
        self.agent = agent
        self.rollout_length = rollout_length
        self.path = path

        if not os.path.exists(path):
            os.makedirs(path)

        files = os.listdir(path)
        latest = [f for f in files if f.isnumeric()]
        latest = sorted(latest, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        self.newest_path = f'{path}/{len(latest)}'
        self.latest_path = f'{path}/{len(latest) - 1}' if len(latest) > 0 else None
        self.latest_path = f'{path}/{len(latest) - 1}' if len(latest) > 0 else None

        self.envs = envs
        self.benchmark = Benchmark()

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
                losses = self.agent.update(trajectories)
                episode_reward = sum([sum(trajectory.rewards) for trajectory in trajectories.values()])
                reward_history.append(episode_reward)

                report({"reward": episode_reward})
                print(f"Episode {episode}, Reward: {episode_reward}, Loss: {losses}")

                values = {key: value for key, value in losses.items()}
                values = {**values, "reward": episode_reward}

                save_checkpoint(episode, episode_reward, losses, values)

        except KeyboardInterrupt:
            pass

        writer.close()
        self.save()

        # return average 100 last rewards or
        n = min(100, len(reward_history))
        if n == 0:
            return 0
        return sum(reward_history[-n:]) / n

    def collect_trajectories(self) -> Dict[str, ReplayBuffer]:
        trajectories = {
            env_id: ReplayBuffer()
            for env_id in self.envs
        }
        states = self.envs.state()
        timesteps = {env_id: 0 for env_id in self.envs}

        dones = set()

        for _ in range(self.rollout_length):
            if dones == set(self.envs):
                break

            self.benchmark.start("trainer", "act")
            batch_states = torch.tensor([states[env_id] for env_id in self.envs], dtype=torch.float32).cuda()
            actions, log_prob, _ = self.agent(batch_states)
            self.benchmark.stop("trainer", "act")

            actions = {env_id: action.tolist() for env_id, action in zip(self.envs, actions)}
            log_prob = {env_id: log_prob.tolist() for env_id, log_prob in zip(self.envs, log_prob)}

            self.benchmark.start("trainer", "step")
            response = self.envs.step(actions, timesteps)
            self.benchmark.stop("trainer", "step")

            self.benchmark.start("trainer", "reset")
            for env_id, data in response.items():
                if data.get("timeout", False):
                    snapshot = self.envs.snapshot(env_id)
                    timesteps[env_id] = snapshot["timestep"]
                    continue
                if "error" in data:
                    raise ValueError(f"Error in response: {response}")
                details = data.pop("details")
                timesteps[env_id] = details["timestep"]
                trajectories[env_id].add(action=actions[env_id], log_prob=log_prob[env_id], **data)

                if data['done']:
                    dones.add(env_id)
                states[env_id] = data['state']
            self.benchmark.stop("trainer", "reset")
        self.envs.reset(list(self.envs))
        print(self.benchmark.summary())

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
