import torch
from torch import nn


class Net(nn.Module):
    def __init__(self, num_layers: int, hidden_size: int, input_size: int, output_size: int):
        super().__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.input_layer = nn.Linear(input_size, hidden_size)

        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor):
        x = self.input_layer(x)
        x = self.layers(x)
        x = self.output_layer(x)
        return x

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    @property
    def flop(self):
        input_layer = 2 * self.input_size * self.hidden_size + self.hidden_size
        hidden_layer = 2 * self.hidden_size * self.hidden_size + self.hidden_size
        # ignoring relu
        output_layer = 2 * self.hidden_size * self.output_size + self.output_size
        return input_layer + self.num_layers * hidden_layer + output_layer


def test_net():
    bs = 4
    input_size = 10
    hidden_size = 5
    output_size = 2

    model = Net(3, hidden_size, input_size, output_size)
    x = torch.randn(bs, input_size)
    y = model(x)
    assert y.shape == (bs, output_size)
