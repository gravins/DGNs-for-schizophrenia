import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from rdkit import Chem
import pandas as pd
import itertools


class CompoundData(Data):
    # This class add the features for all the graph
    def __init__(self, x=None, edge_index=None, edge_attr=None, y=None, pos=None, norm=None, face=None, graph_features=None, **kwargs):
        super(CompoundData, self).__init__(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, pos=pos, norm=norm, face=face, **kwargs)
        self.graph_features = graph_features


class CompoundDataset(InMemoryDataset):
    def __init__(self, dataset_paths, bidir=True, root=".", transform=None, pre_transform=None):
        self.p = dataset_paths
        self.inp_dim = None
        self.graph_ft_dim = None
        self.class_weights = None
        self.bidir = bidir
        self.dose = "no-dose" not in dataset_paths[0]
        super(CompoundDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.bidir, self.dose, self.class_weights, self.inp_dim, self.graph_ft_dim = torch.load(self.processed_paths[1])


    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        tmp = self.p[0].replace(".csv", ".pt").split("/")[-1]
        if self.bidir:
            tmp = tmp.replace(".pt", "-bidir.pt")
        return [tmp, "info_"+tmp]

    def download(self):
        pass

    def process(self):

        data_list = []
        self.class_weights = [0, 0, 0]

        df_main = pd.read_csv(self.p[0])
        df_atom = pd.read_csv(self.p[1], index_col=0)
        df_bond = pd.read_csv(self.p[2], index_col=0) if len(self.p) == 3 else None

        ft = df_main.columns
        smile = "CanonicalSmiles"

        for _, row in df_main.iterrows():
            # Extract molecule from smiles representation
            m = Chem.MolFromSmiles(row[smile])

            # Extract edge features
            edge_attr = None
            if df_bond is not None:
                try:
                    attrs = df_bond.loc[row[smile]].values.tolist()
                    if not isinstance(attrs[0], list):
                        attrs = [attrs]
                    if self.bidir:
                        attrs = list(itertools.chain(*zip(attrs, attrs)))
                    edge_attr = torch.tensor(attrs, dtype=torch.float)
                except KeyError:
                    continue

            # Extract attributes for atoms in the molecule
            # NOTE all the features from bonds and atoms in their dataset are indexed by SMILE string
            x = torch.tensor(df_atom.loc[row[smile]].values, dtype=torch.float)

            # Create the graph connectivity matrix in COO format
            # NOTE: Order is irrelevant
            source_nodes = []
            target_nodes = []
            for bond in m.GetBonds():
                source_nodes.append(bond.GetBeginAtomIdx())
                target_nodes.append(bond.GetEndAtomIdx())

                if self.bidir:
                    # The bonds are bidirectional
                    source_nodes.append(bond.GetEndAtomIdx())
                    target_nodes.append(bond.GetBeginAtomIdx())
            edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)

            if not self.dose:
                graph_level_ft = torch.tensor([[]], dtype=torch.float)
            else:
                graph_level_ft = torch.tensor([[row[ft[1]]]], dtype=torch.float)

            if row[ft[-1]] == -1:
                # Decrease Phagocytosis
                y = torch.tensor([[0, 0, 1]], dtype=torch.float)
                self.class_weights[2] += 1
            elif row[ft[-1]] == 1:
                # Increase Phagocytosis
                y = torch.tensor([[1, 0, 0]], dtype=torch.float)
                self.class_weights[0] += 1
            else:
                # No change
                y = torch.tensor([[0, 1, 0]], dtype=torch.float)
                self.class_weights[1] += 1

            data_list.append(CompoundData(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, graph_features=graph_level_ft))

        self.inp_dim = len(data_list[0].x[0])
        self.graph_ft_dim = len(data_list[0].graph_features[0])

        torch.save((self.bidir, self.dose, self.class_weights, self.inp_dim, self.graph_ft_dim), self.processed_paths[1])

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
