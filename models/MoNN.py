import torch
from torch import nn


class MoNN(nn.Module):
    def __init__(self, input_size, hidden_size):

        super(MoNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.lin = nn.ModuleList()

        self.lin.append(nn.Linear(self.input_size, self.hidden_size[0]))
        self.lin.append(nn.LeakyReLU())
        for i in range(1, len(hidden_size)):
            self.lin.append(nn.Linear(self.hidden_size[i-1], self.hidden_size[i]))
            if i != len(hidden_size) - 1:
                self.lin.append(nn.LeakyReLU())

    def forward(self, data):
        x = data.x

        # Compute the forward for the MLP after the embedding
        for layer in self.lin:
            x = layer(x)

        return x

