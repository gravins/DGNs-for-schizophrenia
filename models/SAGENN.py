import torch
from torch_geometric.nn import SAGEConv
from torch_scatter import scatter_mean

class SAGENN(torch.nn.Module):
    def __init__(self, mlp_input_size, hidden_size, conv_size, batchnorm=True):
        super(SAGENN, self).__init__()

        self.hidden_size = hidden_size
        self.conv_size = conv_size

        # Molecule encoder
        self.gnet = torch.nn.ModuleList()

        for i in range(len(self.conv_size) - 1):
            self.gnet.append(SAGEConv(self.conv_size[i], self.conv_size[i+1]))
            self.gnet.append(torch.nn.LeakyReLU())
            if batchnorm:
                self.gnet.append(torch.nn.BatchNorm1d(self.conv_size[i+1]))


        # MLP readout
        self.lin = torch.nn.ModuleList()

        self.lin.append(torch.nn.Linear(mlp_input_size, self.hidden_size[0]))
        self.lin.append(torch.nn.LeakyReLU())
        for i in range(1, len(self.hidden_size)):
            self.lin.append(torch.nn.Linear(self.hidden_size[i - 1], self.hidden_size[i]))
            if i != len(self.hidden_size) - 1:
                self.lin.append(torch.nn.LeakyReLU())

    def forward(self, data):
        x, edge_index, graph_ft, batch = data.x, data.edge_index, data.graph_features, data.batch

        # Compute embedding for molecules through graph convolution
        for i, g_layer in enumerate(self.gnet):
            if 'Conv' in g_layer.__repr__():
                x = g_layer(x, edge_index) # Graph convolution
            else:
                x = g_layer(x) # Activation function

        x = scatter_mean(x, batch, dim=0)

        # Concatenate features at graph level
        x = torch.cat([x, graph_ft], dim=1)

        # Compute the forward for the MLP after the embedding
        for layer in self.lin:
            x = layer(x)

        return x
 
