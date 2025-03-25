import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .model import layer_init, paramclass, Model


@paramclass
class MMPolicyParams:
    in_features: int
    in_depth: int
    hidden_features: tuple = (128, 128)
    hidden_dims: tuple = (128, 128)
    attention_heads: int = 8
    out_dims: tuple = (4,)


class MMPolicyNetwork(Model):
    def __init__(self,
                 in_features,
                 in_depth,
                 hidden_features=(128, 128),
                 attention_heads=4,
                 hidden_dims=(128, 128),
                 dropout_prob=0.2,
                 out_dims=(4,),
                 num_layers=2
                 ):
        super(MMPolicyNetwork, self).__init__()

        self.in_features = in_features
        self.in_depth = in_depth
        self.hidden_features = hidden_features
        self.attention_heads = attention_heads
        self.hidden_dims = hidden_dims
        self.out_dims = out_dims

        # Number of LOB feature pairs (bid/ask price and quantity at each level)
        lob_feature_dim = 2 * in_depth * 2  # (bid price, bid quantity, ask price, ask quantity)

        indicator_dim = in_features

        ### LOB Feature Extractor (Transformer Encoder)
        self.lob_embedding = nn.Linear(lob_feature_dim, hidden_features[0])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_features[0],
            nhead=attention_heads,
            dim_feedforward=hidden_features[1],
            dropout=dropout_prob,
            activation="relu",
            batch_first=True
        )
        self.lob_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        ### Indicator Feature Extractor (Dense + ReLU)
        self.indicator_mlp = nn.Sequential(
            layer_init(nn.Linear(indicator_dim, hidden_features[0])),
            nn.ReLU(),
            nn.LayerNorm(hidden_features[0], eps=1e-6),
            nn.Dropout(p=dropout_prob),
            *[
                nn.Sequential(
                    layer_init(nn.Linear(dim1, dim2)),
                    nn.ReLU(),
                    nn.LayerNorm(dim2, eps=1e-6),
                    nn.Dropout(p=dropout_prob)
                ) for dim1, dim2 in zip(hidden_features[:-1], hidden_features[1:])
            ]
        )

        ### Fusion + Post-Concatenation MLP
        self.fusion_mlp = nn.Sequential(
            layer_init(nn.Linear(hidden_features[-1] * 2, hidden_dims[0])),
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
        indicators = x[:, :self.in_features]  # First part: market indicators
        lob_features = x[:, self.in_features:]  # Second part: LOB price and quantity pairs

        indicators_out = self.indicator_mlp(indicators)

        lob_embedded = self.lob_embedding(torch.tanh(lob_features))
        lob_out = self.lob_transformer(lob_embedded.unsqueeze(1)).squeeze(1)  # Transformer needs a seq dim

        fused_features = torch.cat([indicators_out, lob_out], dim=-1)

        shared_features = self.fusion_mlp(fused_features)

        outputs = [output_layer(shared_features) for output_layer in self.output_layers]
        return outputs

    def act(self, state):
        probs = self(state)
        dist = [Categorical(prob) for prob in probs]
        action = [d.sample() for d in dist]
        return torch.stack(action, dim=1), [d.log_prob(a) for d, a in zip(dist, action)], dist

    def get_action(self, actions):
        return np.array([a.item() for a in actions])

    def params(self) -> MMPolicyParams:
        return MMPolicyParams(
            in_features=self.in_features,
            in_depth=self.in_depth,
            hidden_features=self.hidden_features,
            hidden_dims=self.hidden_dims,
            attention_heads=self.attention_heads,
            out_dims=self.out_dims
        )


# region sanity checks

def shape_sanity():
    batch_size = 5
    in_features = 3  # Number of market indicators
    in_depth = 10  # Number of LOB levels
    attention_heads = 4
    hidden_features = (128, 128)
    hidden_dims = (128, 128)
    out_dims = [2, 3, 4]  # Multi-head outputs

    model = MMPolicyNetwork(
        in_features=in_features,
        in_depth=in_depth,
        hidden_features=hidden_features,
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
    hidden_features = (128, 128)
    hidden_dims = (128, 128)
    out_dims = [4]  # Single-head test for simplicity

    model = MMPolicyNetwork(
        in_features=in_features,
        in_depth=in_depth,
        hidden_features=hidden_features,
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

    def optim():
        for _ in range(10):
            optimizer.zero_grad()
            actions, log_probs, dist = model.act(x)
            # sum list of tensors into a single tensor
            log_probs = torch.stack(log_probs, dim=1).sum(dim=1)
            loss = torch.mean(log_probs)
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())  # Store loss for tracking

    optim()
    # Check weight updates
    new_weights = model.output_layers[0][0].weight
    assert not torch.allclose(initial_weights, new_weights), "Weights did not change after backpropagation"

    # take moving average
    loss_history = np.array(loss_history)
    loss_history = np.convolve(loss_history, np.ones(3) / 3, mode='valid')

    try:
        assert loss_history[0] > loss_history[-1], "Loss did not decrease in initial iterations"
    except AssertionError:
        print(loss_history)
        loss_history = []
        optim()

    assert loss_history[0] > loss_history[-1], "Loss did not decrease over iterations"

    print("✅ backward_sanity passed! Loss decreased over training.")


def test_save_load():
    params = MMPolicyParams(
        in_features=3,
        in_depth=10,
        attention_heads=4,
        hidden_dims=(128, 128),
        out_dims=(4,)
    )

    model = MMPolicyNetwork.from_params(params)
    checkpoint = model.checkpoint()

    model2 = MMPolicyNetwork.from_checkpoint(checkpoint)

    print("✅ test_save_load passed!")

# endregion

if __name__ == "__main__":
    shape_sanity()
    backward_sanity()
    test_save_load()
    print("All tests passed")
