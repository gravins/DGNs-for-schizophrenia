import torch
from torch_geometric.nn import NNConv
from torch_scatter import scatter_mean
from torch import nn


class ENN(torch.nn.Module):
    def __init__(self, inp, num_edge_feat, extra_inp, conv_size, hidden_size, batchnorm=True):
        super(ENN, self).__init__()
        self.inp = inp
        self.num_ft = num_edge_feat
        self.hidden_size = hidden_size
        self.conv_size = conv_size
        self.extra_inp = extra_inp

        # Molecule encoder
        self.net = nn.ModuleList()

        nn_inner_dim = 2 ** self.num_ft ## this is equal to 2 ** 6
        NN = nn.Sequential(nn.Linear(self.num_ft, nn_inner_dim),
                           nn.LeakyReLU(),
                           nn.Linear(nn_inner_dim, self.inp * self.conv_size[0]))

        self.net.append(NNConv(self.inp, self.conv_size[0], NN, aggr='mean', root_weight=False))
        self.net.append(nn.LeakyReLU())
        if batchnorm: self.net.append(nn.BatchNorm1d(self.conv_size[0]))
        for i in range(1, len(self.conv_size)):
            NN = nn.Sequential(nn.Linear(self.num_ft, nn_inner_dim),
                               nn.LeakyReLU(),
                               nn.Linear(nn_inner_dim, self.conv_size[i-1] * self.conv_size[i]))
            self.net.append(NNConv(self.conv_size[i-1], self.conv_size[i], NN, aggr='mean', root_weight=False))
            self.net.append(nn.LeakyReLU())
            if batchnorm: self.net.append(nn.BatchNorm1d(self.conv_size[i]))

        # MLP readout
        self.net.append(nn.Linear(self.conv_size[-1] + extra_inp, self.hidden_size[0]))
        self.net.append(nn.LeakyReLU())
        for i in range(len(self.hidden_size[:-1])):
            self.net.append(nn.Linear(self.hidden_size[i], self.hidden_size[i+1]))
            if i != len(self.hidden_size) - 1:
                self.net.append(nn.LeakyReLU())

    def forward(self, data):
        x, edge_index, edge_attr, graph_ft, batch = data.x, data.edge_index, data.edge_attr, data.graph_features, data.batch

        for i in range(len(self.net)):
            if 'NNConv' in self.net[i].__repr__():
                x = self.net[i](x, edge_index, edge_attr)
            elif "ReLU" in self.net[i].__repr__():
                x = self.net[i](x)
            elif "Linear" in self.net[i].__repr__():
                if 'NNConv' in self.net[i-2].__repr__() or 'BatchNorm' in self.net[i-1].__repr__():
                    x = scatter_mean(x, batch, dim=0)
                    x = torch.cat([x, graph_ft], dim=1)
                x = self.net[i](x)

        return x
 
