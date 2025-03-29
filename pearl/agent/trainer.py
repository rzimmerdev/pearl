import json
import os
from typing import Dict

import torch
from torch.utils.tensorboard import SummaryWriter

from pearl.agent.agent import Agent
from pearl.agent.replay_buffer import ReplayBuffer
from pearl.envs.multi.env_interface import EnvInterface
from pearl.lib.benchmark import Benchmark


class RLTrainer:
    def __init__(self, agent: Agent, envs: EnvInterface, path="runs", rollout_length=2048):
        super().__init__()
        self.agent = agent
        self.rollout_length = rollout_length
        self.path = path

        try:
            os.makedirs(path)
        except FileExistsError:
            pass

        files = os.listdir(path)
        latest = [f for f in files if f.isnumeric()]
        latest = sorted(latest, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        self.newest_path = f'{path}/{len(latest)}'
        self.latest_path = f'{path}/{len(latest) - 1}' if len(latest) > 0 else None
        self.latest_path = f'{path}/{len(latest) - 1}' if len(latest) > 0 else None

        self.envs = envs
        self.benchmark = Benchmark()

    def checkpoint(self, store_checkpoint=False, writer=None):
        if store_checkpoint and writer is not None:
            def _checkpoint(i, reward, loss, values):
                if i % 10 == 0:
                    print(f'Episode {i}, Reward: {reward}, Loss: {loss}')
                    if i > 0:
                        try:
                            os.remove(f'{self.newest_path}/checkpoint_{i - 10}.pth')
                        except FileNotFoundError:
                            pass

                for key, value in values.items():
                    writer.add_scalar(key, value, i)
            return _checkpoint
        else:
            def _checkpoint(i, reward, loss, values):
                pass
            return _checkpoint

    def train(self, num_episodes=1000, report=None, checkpoint=False):
        reward_history = []

        try:
            os.makedirs(self.newest_path)
        except FileExistsError:
            pass

        if report is None:
            def report(*args, **kwargs):
                pass

        writer = SummaryWriter(log_dir=self.newest_path)
        checkpoint = self.checkpoint(checkpoint, writer)

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

                checkpoint(episode, episode_reward, losses, values)

        except KeyboardInterrupt:
            pass

        writer.close()

        return reward_history

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
                if data.get("timeout", False) or data.get("busy", False):
                    # if busy the environment is very likely currently being step'd
                    # so whatever was sent will likely not be processed for the agent's sake (to save the agent from adverse lag)
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
