import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

from src.agent.model import PolicyNetwork, MultiHeadPolicyNetwork, MMPolicyNetwork, MLP
from src.agent.replay_buffer import ReplayBuffer


def default_reshape(action):
    return action


class PPOAgent:
    def __init__(
            self,
            num_features,
            num_depth,
            action_dim,
            policy_hidden_dims=(128, 128),
            value_hidden_dims=(128, 128),
            attention_heads=8,
            action_reshape=default_reshape,
            gamma=0.99,
            eps_clip=0.2,
            gae_lambda=0.95,
            entropy_coef=0.01,
            lr=1e-3,
            batch_size=32
    ):
        """
        Initialize the PPO agent.

        Args:
            num_features: The number of features in the state space.
            num_depth: The depth of the state space.
            action_dim: The dimension of the action space.
            policy_hidden_dims (tuple): The dimensions of the hidden layers in the policy network.
            value_hidden_dims (tuple): The dimensions of the hidden layers in the value network.
            attention_heads (int): The number of attention heads in the policy network.
            action_reshape (function): A function to reshape the action output from the policy network.
            gamma (float): Discount factor for returns.
            eps_clip (float): Clip parameter used in loss calculation.
            gae_lambda (float): GAE coefficient, affects the balance between bias and variance.
            entropy_coef (float): Entropy coefficient, affects the exploration-exploitation trade-off.
        """
        self.policy_network = MMPolicyNetwork(
            in_features=num_features,
            in_depth=num_depth,
            hidden_dims_features=(policy_hidden_dims[0], policy_hidden_dims[0]),
            attention_heads=attention_heads,
            hidden_dims=policy_hidden_dims,
            out_dims=action_dim
        ).cuda()
        self.value_network = MLP(num_features + 4 * num_depth, 1, hidden_units=value_hidden_dims).cuda()

        self.action_reshape = action_reshape
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.gae_lambda = gae_lambda
        self._entropy_coef = entropy_coef
        self.entropy_coef = entropy_coef

        self.lr = lr
        self.batch_size = batch_size

    def parameters(self):
        # return list(self.policy_network.parameters()) + list(self.value_network.parameters())
        # separate learning rates for policy and value networks
        lr_policy, lr_value = self.lr if isinstance(self.lr, list) else [self.lr, self.lr]
        return [
            {'params': self.policy_network.parameters(), 'lr': lr_policy},
            {'params': self.value_network.parameters(), 'lr': lr_value}
        ]

    def compute_gae(self, rewards, values, dones):
        advantages = []
        advantage = 0
        value = values.tolist()
        value = [value] if isinstance(value, float) else value
        value.append(0)
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * value[t + 1] * (1 - dones[t]) - value[t]
            advantage = delta + self.gamma * self.gae_lambda * advantage * (1 - dones[t])
            advantages.insert(0, advantage)
        return advantages, [adv + val for adv, val in zip(advantages, value[:-1])]

    def collect_trajectories(self, env, rollout_length=2048):
        trajectory = ReplayBuffer()
        state, _ = env.reset()

        for _ in range(rollout_length):
            action, log_prob, _ = self.policy_network.act(torch.tensor(state, dtype=torch.float32).cuda().unsqueeze(0))
            next_state, reward, done, trunc, _ = env.step(self.action_reshape(self.policy_network.get_action(action)))
            done = done or trunc
            trajectory.add(state, action, log_prob, reward, done)
            state = next_state
            if done:
                break

        return trajectory

    def __call__(self, *args, **kwargs):
        return self.policy_network.act(*args, **kwargs)

    def update(self, trajectories, optimizer, epochs=5):
        states = torch.tensor(np.array(trajectories.states), dtype=torch.float32).cuda()
        actions = torch.tensor(trajectories.actions).cuda()
        old_log_probs = torch.tensor(trajectories.log_probs, dtype=torch.float32).cuda()
        rewards = torch.tensor(trajectories.rewards, dtype=torch.float32).cuda()
        dones = torch.tensor(trajectories.dones, dtype=torch.float32).cuda()

        state_values = self.value_network(states).squeeze().detach().cpu().numpy()
        advantages, returns = self.compute_gae(rewards.cpu().numpy(), state_values, dones.cpu().numpy())

        advantages = torch.tensor(advantages, dtype=torch.float32).cuda()
        returns = torch.tensor(returns, dtype=torch.float32).cuda()

        dataset = torch.utils.data.TensorDataset(states, actions, old_log_probs, advantages, returns)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        ep_actor_loss = 0
        ep_critic_loss = 0

        for epoch in range(epochs):  # Number of training epochs
            for batch_states, batch_actions, batch_old_log_probs, batch_advantages, batch_returns in data_loader:
                _, _, dist = self.policy_network.act(batch_states)
                log_probs = self.logprobs(dist, batch_actions)
                entropy = self.entropy(dist)

                ratios = torch.exp(log_probs.sum(dim=-1) - batch_old_log_probs.sum(dim=-1))

                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                state_values = self.value_network(batch_states).squeeze()

                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
                critic_loss = nn.MSELoss()(state_values, batch_returns)
                # loss = actor_loss + 0.5 * critic_loss

                optimizer.zero_grad()
                actor_loss.backward()
                optimizer.step()

                optimizer.zero_grad()
                critic_loss.backward()
                optimizer.step()

                # episode_loss += loss.item()
                ep_actor_loss += actor_loss.item()
                ep_critic_loss += critic_loss.item()

        self.entropy_coef *= 0.999

        return {
            "actor_loss": ep_actor_loss / epochs,
            "critic_loss": ep_critic_loss / epochs
        }

    def reset(self):
        self.entropy_coef = self._entropy_coef

    def logprobs(self, dists, actions):
        if isinstance(dists, list):
            return torch.stack([dist.log_prob(act) for dist, act in zip(dists, actions.T)]).T
        return dists.log_prob(actions)

    def entropy(self, dists):
        if isinstance(dists, list):
            return torch.stack([dist.entropy() for dist in dists]).mean()
        return dists.entropy().mean()

    def save_weights(self, path):
        torch.save({
            'policy_state_dict': self.policy_network.state_dict(),
            'value_state_dict': self.value_network.state_dict()
        }, path)

    def load_weights(self, path):
        checkpoint = torch.load(path, weights_only=True)
        self.policy_network.load_state_dict(checkpoint['policy_state_dict'])
        self.value_network.load_state_dict(checkpoint['value_state_dict'])
