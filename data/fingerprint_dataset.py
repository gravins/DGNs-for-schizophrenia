import torch
from torch_geometric.data import InMemoryDataset
from .compound_dataset import CompoundData
import pandas as pd


class FingerprintDataset(InMemoryDataset):
    def __init__(self, dataset_paths, root=".", transform=None, pre_transform=None):
        self.p = dataset_paths
        self.class_weights = None
        super(FingerprintDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return self.p.replace(".csv", ".pt").split("/")[-1]

    def download(self):
        pass

    def process(self):
        data_list = []
        self.class_weights = [0, 0, 0]

        df = pd.read_csv(self.p)

        ft = df.columns

        for _, row in df.iterrows():
            if len(ft) > int(self.p.split("L")[1].split("-")[0]) + 2:
                ### x_1, x_2, ...., dose, n_atoms, y
                g = [df.columns[-2], df.columns[-3]]
                group = torch.tensor([row[g].values], dtype=torch.float)
                
                # Get molecule's fingerprint
                x = torch.tensor([row[ft[:-2]].values], dtype=torch.float)
            else:
                # Get molecule's fingerprint
                x = torch.tensor([row[ft[:-1]].values], dtype=torch.float)

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

            data_list.append(CompoundData(x=x, y=y) if len(ft) == int(self.p.split("L")[1].split("-")[0]) + 2 else CompoundData(x=x, y=y, graph_features=group))

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
