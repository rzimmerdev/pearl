import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Layer initialization helper.
    """
    nn.init.xavier_uniform_(layer.weight, gain=std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_units=(64, 64), hidden_activation=nn.ReLU(), output_activation=None):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(hidden_units)):
            if i == 0:
                self.layers.append(layer_init(nn.Linear(in_dim, hidden_units[i])))
            else:
                self.layers.append(layer_init(nn.Linear(hidden_units[i - 1], hidden_units[i])))
            self.layers.append(hidden_activation)
        self.layers.append(layer_init(nn.Linear(hidden_units[-1], out_dim)))
        if output_activation:
            self.layers.append(output_activation)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ValueNetwork(nn.Module):
    def __init__(self, in_dim):
        super(ValueNetwork, self).__init__()
        self.layers = nn.Sequential(
            layer_init(nn.Linear(in_dim, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 1), std=1)
        )

    def forward(self, x):
        return self.layers(x)


class PolicyNetwork(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(PolicyNetwork, self).__init__()
        self.layers = nn.Sequential(
            layer_init(nn.Linear(in_dim, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, out_dim), std=0.01),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.layers(x)

    def act(self, state):
        action_probs = self(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action, dist.log_prob(action), dist

    def get_action(self, action):
        return action.item()


class MultiHeadPolicyNetwork(nn.Module):
    def __init__(
            self,
            in_dim,
            out_dims,
            hidden_dims=(128, 128),
            dropout_prob=0.2
    ):
        super(MultiHeadPolicyNetwork, self).__init__()

        self.shared_layers = nn.Sequential(
            layer_init(nn.Linear(in_dim, hidden_dims[0])),
            nn.ReLU(),
            nn.LayerNorm(hidden_dims[0]),  # Use LayerNorm instead of BatchNorm1d
            nn.Dropout(p=dropout_prob),
            *[
                nn.Sequential(
                    layer_init(nn.Linear(dim1, dim2)),
                    nn.ReLU(),
                    nn.LayerNorm(dim2),  # Use LayerNorm instead of BatchNorm1d
                    nn.Dropout(p=dropout_prob)
                ) for dim1, dim2 in zip(hidden_dims[:-1], hidden_dims[1:])
            ]
        )

        self.output_layers = nn.ModuleList([
            nn.Sequential(
                layer_init(nn.Linear(hidden_dims[-1], out_dim), std=0.01),
                nn.Softmax(dim=-1)
            ) for out_dim in out_dims
        ])

    def forward(self, x):
        shared_features = self.shared_layers(x)
        outputs = [output_layer(shared_features) for output_layer in self.output_layers]
        return outputs

    def act(self, state):
        probs = self(state)
        dist = [Categorical(prob) for prob in probs]
        action = [d.sample() for d in dist]
        return action, [d.log_prob(a) for d, a in zip(dist, action)], dist

    def get_action(self, actions):
        return np.array([a.item() for a in actions])


class MMPolicyNetwork(nn.Module):
    def __init__(
            self,
            in_features,
            in_depth,
            hidden_dims_features=(128, 128),
            attention_heads=4,
            hidden_dims=(128, 128),
            dropout_prob=0.2,
            out_dims=(4,),
            num_layers=2
    ):
        super(MMPolicyNetwork, self).__init__()

        self.in_features = in_features
        self.in_depth = in_depth

        # Number of LOB feature pairs (bid/ask price and quantity at each level)
        lob_feature_dim = 2 * in_depth * 2  # (bid price, bid quantity, ask price, ask quantity)

        # Number of market indicators
        indicator_dim = in_features

        ### LOB Feature Extractor (Transformer Encoder)
        self.lob_embedding = nn.Linear(lob_feature_dim, hidden_dims_features[0])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dims_features[0],
            nhead=attention_heads,
            dim_feedforward=hidden_dims_features[1],
            dropout=dropout_prob,
            activation="relu",
            batch_first=True
        )
        self.lob_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        ### Indicator Feature Extractor (Dense + ReLU)
        self.indicator_mlp = nn.Sequential(
            layer_init(nn.Linear(indicator_dim, hidden_dims_features[0])),
            nn.ReLU(),
            nn.LayerNorm(hidden_dims_features[0], eps=1e-6),
            nn.Dropout(p=dropout_prob),
            *[
                nn.Sequential(
                    layer_init(nn.Linear(dim1, dim2)),
                    nn.ReLU(),
                    nn.LayerNorm(dim2, eps=1e-6),
                    nn.Dropout(p=dropout_prob)
                ) for dim1, dim2 in zip(hidden_dims_features[:-1], hidden_dims_features[1:])
            ]
        )

        ### Fusion + Post-Concatenation MLP
        self.fusion_mlp = nn.Sequential(
            layer_init(nn.Linear(hidden_dims_features[-1] * 2, hidden_dims[0])),
            nn.ReLU(),
            nn.LayerNorm(hidden_dims[0], eps=1e-6),
            nn.Dropout(p=dropout_prob),
            *[
                nn.Sequential(
                    layer_init(nn.Linear(dim1, dim2)),
                    nn.ReLU(),
                    nn.LayerNorm(dim2, eps=1e-6),
                    nn.Dropout(p=dropout_prob)
                ) for dim1, dim2 in zip(hidden_dims[:-1], hidden_dims[1:])
            ]
        )

        ### Multi-Head Output Layers
        self.output_layers = nn.ModuleList([
            nn.Sequential(
                layer_init(nn.Linear(hidden_dims[-1], out_dim), std=0.01),
                nn.Softmax(dim=-1)  # For categorical actions (if using Gaussian, replace with Tanh or Linear)
            ) for out_dim in out_dims
        ])

    def forward(self, x):
        # Split input into market indicators and LOB features
        indicators = x[:, :self.in_features]  # First part: market indicators
        lob_features = x[:, self.in_features:]  # Second part: LOB price and quantity pairs

        # Process indicators through MLP
        indicators_out = self.indicator_mlp(indicators)

        # Process LOB features through Transformer
        lob_embedded = self.lob_embedding(torch.tanh(lob_features))
        lob_out = self.lob_transformer(lob_embedded.unsqueeze(1)).squeeze(1)  # Transformer needs a seq dim

        # Concatenate outputs from both paths
        fused_features = torch.cat([indicators_out, lob_out], dim=-1)

        # Pass through final MLP
        shared_features = self.fusion_mlp(fused_features)

        # Output layers (multi-head policy network)
        outputs = [output_layer(shared_features) for output_layer in self.output_layers]
        return outputs

    def act(self, state):
        probs = self(state)
        dist = [Categorical(prob) for prob in probs]
        action = [d.sample() for d in dist]
        return action, [d.log_prob(a) for d, a in zip(dist, action)], dist

    def get_action(self, actions):
        return np.array([a.item() for a in actions])


def shape_sanity():
    batch_size = 5
    in_features = 3  # Number of market indicators
    in_depth = 10  # Number of LOB levels
    attention_heads = 4
    hidden_dims_features = (128, 128)
    hidden_dims = (128, 128)
    out_dims = [2, 3, 4]  # Multi-head outputs

    model = MMPolicyNetwork(
        in_features=in_features,
        in_depth=in_depth,
        hidden_dims_features=hidden_dims_features,
        attention_heads=attention_heads,
        hidden_dims=hidden_dims,
        out_dims=out_dims
    )

    in_dim = in_features + (2 * in_depth * 2)  # Indicator size + LOB size
    x = torch.randn(batch_size, in_dim)

    probs = model(x)
    assert len(probs) == len(out_dims), "Number of outputs does not match out_dims"
    for i, prob in enumerate(probs):
        assert prob.shape == (
            batch_size,
            out_dims[i]), f"Output {i} shape mismatch: expected {(batch_size, out_dims[i])}, got {prob.shape}"

    actions, log_probs, dist = model.act(x)
    assert len(actions) == len(out_dims), "Number of sampled actions mismatch"
    assert len(log_probs) == len(out_dims), "Number of log probabilities mismatch"
    for i, action in enumerate(actions):
        assert action.shape == (batch_size,), f"Action {i} shape mismatch: expected {(batch_size,)}, got {action.shape}"

    print("✅ shape_sanity passed!")


def backward_sanity():
    batch_size = 5
    in_features = 3
    in_depth = 10
    attention_heads = 4
    hidden_dims_features = (128, 128)
    hidden_dims = (128, 128)
    out_dims = [4]  # Single-head test for simplicity

    model = MMPolicyNetwork(
        in_features=in_features,
        in_depth=in_depth,
        hidden_dims_features=hidden_dims_features,
        attention_heads=attention_heads,
        hidden_dims=hidden_dims,
        out_dims=out_dims
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Create input and target
    in_dim = in_features + (2 * in_depth * 2)
    x = torch.randn(batch_size, in_dim)
    y = torch.randint(0, out_dims[0], (batch_size,))

    # Store initial weights of the last layer
    initial_weights = model.output_layers[0][0].weight.clone().detach()

    # Track loss values
    loss_history = []

    for _ in range(10):
        optimizer.zero_grad()
        actions, log_probs, dist = model.act(x)
        loss = -sum(log_probs).mean()  # PPO-like objective (negative log probability)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())  # Store loss for tracking

    # Check weight updates
    new_weights = model.output_layers[0][0].weight
    assert not torch.allclose(initial_weights, new_weights), "Weights did not change after backpropagation"

    assert loss_history[0] > loss_history[-1], "Loss did not decrease over iterations"

    print("✅ backward_sanity passed! Loss decreased over training.")


if __name__ == "__main__":
    shape_sanity()
    backward_sanity()
    print("All tests passed")
