from sklearn.model_selection import StratifiedKFold
from torch.utils.data.dataset import random_split, ConcatDataset
from ._split import *
import numpy as np
import random
import torch


def stratify_split(dataset, test_size=0.2, shuffle=False, random_seed=None):

    random.seed(random_seed)

    classes = {}
    for i, data in enumerate(dataset):
        if str(data.y) not in classes.keys():
            classes[str(data.y)] = [i]
        else:
            classes[str(data.y)].append(i)

    if shuffle:
        for k in classes.keys():
            random.shuffle(classes[k])

    tr_indexes = []
    ts_indexes = []
    for k in classes.keys():
        ts_indexes += (classes[k][:int(len(classes[k]) * test_size)])
        tr_indexes += (classes[k][int(len(classes[k]) * test_size):])

    return tr_indexes, ts_indexes


def make_k_fold_split(dataset, k, stratify=True, group=None, shuffle=True, random_seed=None):
    """
    output format:
        [(tr_1, val_1), ... , (tr_k, val_k)]
    """
    if random_seed is not None:
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)

    train_val = []
    if stratify:
        if group is None:
            s = StratifiedKFold(n_splits=k, shuffle=shuffle, random_state=random_seed)
            for train_index, val_index in s.split([[1]] * len(dataset), [torch.argmax(td.y, 1) for td in dataset]):
                train_val.append((dataset[torch.tensor(train_index)], dataset[torch.tensor(val_index)]))
        else:
            s = DAStratifiedKFold(n_splits=k, random_state=random_seed, shuffle=shuffle)
            for train_index, val_index in s.split([[1]] * len(dataset), [torch.argmax(td.y, 1) for td in dataset], group):
                train_val.append((dataset[torch.tensor(train_index)], dataset[torch.tensor(val_index)]))
    else:
        folds_dim = len(dataset) / k
        folds_len = [int(folds_dim * (i + 1)) - int(folds_dim * i) for i in range(k)]

        if shuffle:
            dataset = dataset.shuffle()
        for i in range(k):
            start_val = sum(folds_len[:i])
            val = dataset[torch.tensor(range(start_val, start_val + folds_len[i]))]
            train = torch.cat([dataset[:start_val], start_val + dataset[folds_len[i]:]])
            train_val.append((train, val))

    return train_val

'''
def standardise(dataset, column_id):
    l = []
    for data in dataset:
        l += data.x.tolist()

    df = pd.DataFrame.from_records(l)
    mean = df[column_id].mean()
    std = df[column_id].std()
    df[column_id] = (df[column_id] - mean) / std

    j = 0
    for i, data in enumerate(dataset):

        xlen = len(data.x)
        a = []
        for _ in range(xlen):
            a.append(df.iloc[j].values)
            j += 1

        print(dataset[i].__setitem__("x", torch.FloatTensor(a)))
        exit()
    print(j - len(df))
    print(dataset[0].x)
    exit()
    return dataset, mean, std

'''

'''
def make_nested_cv_split(dataset, inner, outer, random_seed=None):
    """
    output format:
        [   ([(tr_1, val_1), ... , (tr_inner, val_inner)]_1 , ts_1),

            ...

            ([(tr_1, val_1), ... , (tr_inner, val_inner)]_outer , ts_outer)  ]
    """

    train_test = make_k_fold_split(dataset, outer, random_seed)

    train_val_test = []
    for train, test in train_test:
        train_val_test.append(make_k_fold_split(train, inner, random_seed))

    return train_val_test
'''