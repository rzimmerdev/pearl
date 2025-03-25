from torch import nn

from .model import layer_init, paramclass, Model


@paramclass
class MLPParams:
    in_dim: int
    out_dim: int
    hidden_units: tuple = (64, 64)
    hidden_activation: nn.Module = nn.ReLU()


class MLP(Model[MLPParams]):
    def __init__(self, in_dim, out_dim, hidden_units=(64, 64), hidden_activation: nn.Module = nn.ReLU(),
                 output_activation=None):
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

    def params(self) -> MLPParams:
        return MLPParams(
            in_dim=self.in_dim,
            out_dim=self.out_dim,
            hidden_units=self.hidden_units,
            hidden_activation=self.hidden_activation
        )
