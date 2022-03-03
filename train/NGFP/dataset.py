import torch as T
from torch.utils.data import Dataset
from . import preprocessing as prep
from copy import deepcopy


class MolData(Dataset):
    """Custom PyTorch Dataset that takes a file containing \n separated SMILES"""
    def __init__(self, smiles, labels, graph):
        self.max_atom = 80
        self.max_degree = 6
        print(self)
        self.atoms, self.bonds, self.edges = self._featurize(smiles)
        self.label = T.from_numpy(labels).float()
        self.graph = T.from_numpy(graph).float()


    def _featurize(self, smiles):
        return prep.tensorise_smiles(smiles, max_atoms=self.max_atom, max_degree=self.max_degree)

    def __getitem__(self, i):
        if isinstance(i, int):
            return self.atoms[i], self.bonds[i], self.edges[i], self.label[i], self.graph[i]
        if isinstance(i, list):
            return self.atoms[i], self.bonds[i], self.edges[i], self.label[i], self.graph[i]
        elif T.is_tensor(i) and i.dtype == T.long:
            return self.__indexing__(i)
        else:
            raise TypeError("Argument i can be int or torch.long tensor, not ", type(i))

    def __indexing__(self, index):
        obj_copy = self.__class__.__new__(self.__class__)
        obj_copy.__dict__ = deepcopy(self.__dict__)
        a, b, e, l, g = self.__getitem__(index.tolist())
        obj_copy.atoms = a
        obj_copy.bonds = b
        obj_copy.edges = e
        obj_copy.label = l
        obj_copy.graph = g
        return obj_copy

    def split(self, batch_size):
        return

    def __len__(self):
        return len(self.label)

