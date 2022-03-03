import torch
from torch import nn
from collections import OrderedDict
import torch_scatter


class LinNN(nn.Module):
    def __init__(self, mlp_input_size, hidden_size, embedding_size=[], agg='mean'):
        super(LinNN, self).__init__()

        assert agg == 'mean' or agg == 'sum', f'agg can be mean or sum, not {agg}'

        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.embedding = None
        self.mlp_inp = mlp_input_size
        self.agg = agg

        self.lin = nn.ModuleList()

        # Atom encoder
        if len(self.embedding_size) > 1:
            seq = []
            for i in range(1, len(self.embedding_size)):
                seq.append(("emb_lin" + str(i-1), nn.Linear(self.embedding_size[i-1], self.embedding_size[i])))
                if i == len(self.embedding_size) - 1:
                    seq.append(("emb_act" + str(i - 1), nn.Tanh()))
                else:
                    seq.append(("emb_act" + str(i-1), nn.LeakyReLU()))

            self.embedding_net = nn.Sequential(OrderedDict(seq))
            self.lin.append(self.embedding_net)

        # MLP readout
        self.lin.append(nn.Linear(self.mlp_inp, self.hidden_size[0]))
        for i in range(1, len(hidden_size)):
            self.lin.append(nn.Linear(self.hidden_size[i-1], self.hidden_size[i]))

    def forward(self, data):
        x, batch = data.x, data.batch
        idx = 0
        
        # Compute embedding for each node and sum nodes of same graph
        if len(self.embedding_size) > 0 or self.embedding is not None:
            if self.agg == 'mean':
                x = torch_scatter.scatter_mean(self.lin[0](x), batch, dim=0)  # AVG
            else:
                x = torch_scatter.scatter_add(self.lin[0](x), batch, dim=0)  # SUM
            idx = 1

        # Concatenate features at graph level
        x = torch.cat([x, data.graph_features], dim=1)

        # Compute the forward for the MLP after the embedding
        for layer in self.lin[idx:-1]:
            x = layer(x).relu()
        x = self.lin[-1](x)
        return x
