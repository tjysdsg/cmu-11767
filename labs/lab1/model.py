import torch
from torch import nn


class Net(nn.Module):
    def __init__(self, num_layers: int, hidden_size: int, input_size: int, output_size: int):
        super().__init__()

        self.input_layer = nn.Linear(input_size, hidden_size)

        self.layers = nn.Sequential(*[
            nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)
        ])

        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor):
        x = self.input_layer(x)
        x = self.layers(x)
        x = self.output_layer(x)
        return x


def test_net():
    bs = 4
    input_size = 10
    hidden_size = 5
    output_size = 2

    model = Net(3, hidden_size, input_size, output_size)
    x = torch.randn(bs, input_size)
    y = model(x)
    assert y.shape == (bs, output_size)
