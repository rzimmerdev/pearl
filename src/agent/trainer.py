import json
import os
from typing import Dict

import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from src.agent.agent import Agent
from src.agent.replay_buffer import ReplayBuffer
from src.env.multi.env_interface import EnvInterface


class RLTrainer:
    def __init__(self, agent: Agent, envs: EnvInterface, path="runs", rollout_length=2048):
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
        self.envs = envs

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

        env_ids = set(self.envs)

        for _ in range(self.rollout_length):
            if not env_ids:
                break

            # transform states {env_id: state} into batch x state tensor, where (batch x state)[i] == envs.values()[i]
            batch_states = torch.tensor([states[env_id] for env_id in env_ids], dtype=torch.float32).cuda()
            actions, log_prob, _ = self.agent(batch_states)

            # action is of format [tensor(num_envs) for _ in range(action_dim)], but need [tensor(action_dim) for _ in range(num_envs)]
            # transform back into {env_id: action} according to env_ids order
            actions = {env_id: action.tolist() for env_id, action in zip(env_ids, actions)}
            log_prob = {env_id: log_prob.tolist() for env_id, log_prob in zip(env_ids, log_prob)}

            response = self.envs.step(actions, timesteps)

            # store actions, log_probs, rewards, and next_states in respective ReplayBuffer
            dones = []
            for env_id, data in response.items():
                if data.get("timeout", False):
                    raise TimeoutError(f"Env response timed out {response}")
                if "error" in data:
                    raise ValueError(f"Error in response: {response}")
                details = data.pop("details")
                timesteps[env_id] = details["timestep"]
                trajectories[env_id].add(action=actions[env_id], log_prob=log_prob[env_id], **data)

                # if done 1. call reset, 2. remove from env ids
                if data['done']:
                    dones.append(env_id)
                    env_ids.remove(env_id)
                states[env_id] = data['state']
            self.envs.reset(dones)

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
