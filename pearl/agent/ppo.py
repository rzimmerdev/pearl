from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from .agent import Agent, default_reshape


@dataclass
class PPOParams:
    gamma: float = 0.99
    eps_clip: float = 0.2
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01

    # method for serialization
    def __getitem__(self, item):
        return getattr(self, item)

    def items(self):
        return self.__dict__.items()

    # method for deserialization
    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    def to_dict(self):
        return self.__dict__


class PPOAgent(Agent):
    def __init__(self,
                 policy_network: nn.Module = None,
                 value_network: nn.Module = None,
                 params: PPOParams = PPOParams(),
                 action_reshape=default_reshape,
                 lr=1e-3,
                 batch_size=32):
        """
        Initialize the PPO agent.
        """
        # num_features: The number of features in the state space.
        # num_depth: The depth of the state space.
        # action_dim: The dimension of the action space.
        # policy_hidden_dims (tuple): The dimensions of the hidden layers in the policy network.
        # value_hidden_dims (tuple): The dimensions of the hidden layers in the value network.
        # attention_heads (int): The number of attention heads in the policy network.
        # action_reshape (function): A function to reshape the action output from the policy network.
        # gamma (float): Discount factor for returns.
        # eps_clip (float): Clip parameter used in loss calculation.
        # gae_lambda (float): GAE coefficient, affects the balance between bias and variance.
        # entropy_coef (float): Entropy coefficient, affects the exploration-exploitation trade-off.

        # self.policy_network = MMPolicyNetwork(
        #     in_features=num_features,
        #     in_depth=num_depth,
        #     hidden_dims_features=(policy_hidden_dims[0], policy_hidden_dims[0]),
        #     attention_heads=attention_heads,
        #     hidden_dims=policy_hidden_dims,
        #     out_dims=action_dim
        # ).cuda()
        # self.value_network = MLP(num_features + 4 * num_depth, 1, hidden_units=value_hidden_dims).cuda()
        super().__init__()
        self.policy_network = policy_network
        self.value_network = value_network

        self.action_reshape = action_reshape
        self.params = params

        self.entropy_coef = params.entropy_coef
        self.lr = lr
        self.batch_size = batch_size

        self.optimizer = optim.Adam(
            self.parameters(),
            betas=(0.9, 0.999),
            weight_decay=1e-5,
            amsgrad=True
        )

    def parameters(self):
        # return list(self.policy_network.parameters()) + list(self.value_network.parameters())
        # separate learning rates for policy and value networks
        lr_policy, lr_value = self.lr if isinstance(self.lr, list) else [self.lr, self.lr]
        return [
            {'params': self.policy_network.parameters(), 'lr': lr_policy},
            {'params': self.value_network.parameters(), 'lr': lr_value}
        ]

    @property
    def models(self) -> Dict[str, nn.Module]:
        return {
            'policy_network': self.policy_network,
            'value_network': self.value_network
        }

    def save(self, path):
        torch.save({
            'policy_state_dict': self.policy_network.state_dict(),
            'value_state_dict': self.value_network.state_dict(),
            'params': self.params.to_dict(),
            'lr': self.lr,
            'batch_size': self.batch_size
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.policy_network.load_state_dict(checkpoint['policy_state_dict'])
        self.value_network.load_state_dict(checkpoint['value_state_dict'])
        self.params = PPOParams.from_dict(checkpoint['params'])

    @classmethod
    def from_checkpoint(cls, path):
        checkpoint = torch.load(path)
        policy_network = checkpoint['policy_network']
        value_network = checkpoint['value_network']
        params = PPOParams.from_dict(checkpoint['params'])
        return cls(policy_network, value_network, params, checkpoint['lr'], checkpoint['batch_size'])

    def compute_gae(self, rewards, values, dones):
        advantages = []
        advantage = 0
        value = values.tolist()
        value = [value] if isinstance(value, float) else value
        value.append(0)
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.params.gamma * value[t + 1] * (1 - dones[t]) - value[t]
            advantage = delta + self.params.gamma * self.params.gae_lambda * advantage * (1 - dones[t])
            advantages.insert(0, advantage)
        return advantages, [adv + val for adv, val in zip(advantages, value[:-1])]

    # def collect_trajectories(self, env, rollout_length=2048):
    #     trajectory = ReplayBuffer()
    #
    #     for _ in range(rollout_length):
    #         action, log_prob, _ = self.policy_network.act(torch.tensor(state, dtype=torch.float32).cuda().unsqueeze(0))
    #         next_state, reward, done, trunc, _ = env.send(self.action_reshape(self.policy_network.get_action(action)))
    #         done = done or trunc
    #         trajectory.add(state, action, log_prob, reward, done)
    #         state = next_state
    #         if done:
    #             break
    #
    #     return trajectory

    def __call__(self, *args, **kwargs):
        return self.act(*args, **kwargs)

    def update(self, trajectories, epochs=5):
        # Collect and concatenate data from all environments
        states = []
        actions = []
        old_log_probs = []
        rewards = []
        dones = []

        for env_id, buffer in trajectories.items():
            states.append(np.array(buffer.states))
            actions.append(np.array(buffer.actions))
            old_log_probs.append(np.array(buffer.log_probs))
            rewards.append(np.array(buffer.rewards))
            dones.append(np.array(buffer.dones))

        states = torch.tensor(np.concatenate(states, axis=0), dtype=torch.float32).cuda()
        actions = torch.tensor(np.concatenate(actions, axis=0)).cuda()
        old_log_probs = torch.tensor(np.concatenate(old_log_probs, axis=0), dtype=torch.float32).cuda()
        rewards = torch.tensor(np.concatenate(rewards, axis=0), dtype=torch.float32).cuda()
        dones = torch.tensor(np.concatenate(dones, axis=0), dtype=torch.float32).cuda()

        # Compute advantages and returns
        state_values = self.value_network(states).squeeze().detach().cpu().numpy()
        advantages, returns = self.compute_gae(rewards.cpu().numpy(), state_values, dones.cpu().numpy())

        advantages = torch.tensor(advantages, dtype=torch.float32).cuda()
        returns = torch.tensor(returns, dtype=torch.float32).cuda()

        # Create dataset and data loader
        dataset = torch.utils.data.TensorDataset(states, actions, old_log_probs, advantages, returns)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        ep_actor_loss = 0
        ep_critic_loss = 0
        num_batches = 0

        # Training loop
        for epoch in range(epochs):
            for batch_states, batch_actions, batch_old_log_probs, batch_advantages, batch_returns in data_loader:
                _, _, dist = self.policy_network.act(batch_states)
                log_probs = self.logprobs(dist, batch_actions)
                entropy = self.entropy(dist)

                ratios = torch.exp(log_probs.sum(dim=-1) - batch_old_log_probs.sum(dim=-1))

                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.params.eps_clip, 1 + self.params.eps_clip) * batch_advantages
                state_values = self.value_network(batch_states).squeeze()

                actor_loss = (-torch.min(surr1, surr2).mean() - self.entropy_coef * entropy) / batch_states.shape[0]
                critic_loss = nn.MSELoss()(state_values, batch_returns) / batch_states.shape[0]

                self.optimizer.zero_grad()
                actor_loss.backward()
                self.optimizer.step()

                self.optimizer.zero_grad()
                critic_loss.backward()
                self.optimizer.step()

                ep_actor_loss += actor_loss.item()
                ep_critic_loss += critic_loss.item()
                num_batches += 1
        num_updates = epochs * len(data_loader)

        # Report averaged loss per epoch
        return {
            "actor_loss": ep_actor_loss / (num_batches * epochs),
            "critic_loss": ep_critic_loss / (num_batches * epochs),
        }, num_updates


    def act(self, state):
        return self.policy_network.act(state)

    def reset(self):
        self.entropy_coef = self.params.entropy_coef

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
