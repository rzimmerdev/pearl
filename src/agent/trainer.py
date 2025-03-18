import os

from torch import optim
from torch.utils.tensorboard import SummaryWriter


class RLTrainer:
    def __init__(self, env, agent, lr=1e-4, num_envs=4, path="runs", rollout_length=2048):
        self.env = env
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

    def train(self, num_episodes=1000, report=None, checkpoint=False):
        reward_history = []

        if not os.path.exists(self.newest_path):
            os.makedirs(self.newest_path)

        snapshots = {col: [] for col in self.env.snapshot_columns}
        snapshots["episode"] = []

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
                trajectories = self.agent.collect_trajectories(self.env, self.rollout_length)
                losses = self.agent.update(trajectories, self.optimizer)
                episode_reward = sum(trajectories.rewards)
                reward_history.append(episode_reward)

                report({"reward": episode_reward})

                snapshot = self.env.snapshot()
                snapshot["episode"] = episode

                for col in snapshots.keys():
                    snapshots[col].append(snapshot[col])

                values = {key: value for key, value in losses.items()}
                values = {**values, **snapshot, "reward": episode_reward}

                save_checkpoint(episode, episode_reward, losses, values)

        except KeyboardInterrupt:
            pass

        writer.close()
        self.save()

        # return average 100 last rewards or
        n = min(100, len(reward_history))
        return sum(reward_history[-n:]) / n

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
