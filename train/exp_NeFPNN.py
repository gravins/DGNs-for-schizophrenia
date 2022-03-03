
import torch

from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score, accuracy_score
from .utils import multiclass_roc_auc_score, per_class_accuracy
from sklearn.model_selection import StratifiedKFold
from pathos.multiprocessing import ProcessingPool
from .train_assess import avg_res_validation
from .NGFP.dataset import MolData
from .NGFP.model import NeFPNN
import pandas as pd
from .draw import *
import numpy as np
import itertools
import random
import pickle
import time
import copy


random_seed = 42
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)
random.seed(random_seed)

metrics = [multiclass_roc_auc_score, per_class_accuracy, f1_score, recall_score,
           precision_score, accuracy_score, confusion_matrix]
metric_scorer = "multiclass_roc_auc_score" 

def thread_fun(parameters, loss_fun, batch, epochs, s, j, o, train_dataset, device):
    res = []
    params = parameters[0]
    optimiz = parameters[1]
    folds = s.split([[1]] * len(train_dataset), torch.argmax(train_dataset.label, 1))
    for i, (trained, valided) in enumerate(folds):
          train_set = train_dataset[torch.tensor(trained, dtype=torch.long)]
          valid_set = train_dataset[torch.tensor(valided, dtype=torch.long)]

          net = NeFPNN(params, device).to(device)
          out = 'NeFPNN_' + str(j) + '_' + str(o)
          history, best, best_res = net.fit(train_set, valid_set, metrics=metrics, 
                                            metric_scorer=metric_scorer, 
                                            batch=batch, epochs=epochs, optimizer=optimiz, 
                                            path='%s_%d' % (out, i), criterion=loss_fun)
          res.append((best_res, history))
    tr_avg_metr, tr_std_metr, tr_avg_loss, tr_std_loss, avg_metr, std_metr, avg_loss, std_loss = avg_res_validation(res)
    mod = NeFPNN(params, device)
    top = {"val_avg_loss": avg_loss,
           "val_std_loss": std_loss,
           "val_avg_metr": avg_metr,
           "val_std_metr": std_metr,
           "tr_avg_loss": tr_avg_loss,
           "tr_std_loss": tr_std_loss,
           "tr_avg_metr": tr_avg_metr,
           "tr_std_metr": tr_std_metr,
           "single_res": res,
           "model": mod,
           "optimizer": optimiz}
    return top


def run_NeFPNN(mode, optim_grid, model_params, random_seed, device, args):
    assert mode in ['cv', 'test']

    train_path = './Datasets/NGFP_train_dataset'
    test_path = './Datasets/NGFP_test_dataset'
    
    start = time.time()
    try:
        train_dataset = pickle.load(open(train_path+".p", "rb"))
        test_dataset = pickle.load(open(test_path+".p", "rb"))
    except FileNotFoundError as e:
        train_dataset = pd.read_csv(train_path+".csv")
        test_dataset = pd.read_csv(test_path+".csv")

        ft = train_dataset.columns
        t = lambda x: [0, 0, 1] if x == -1 else [1, 0, 0] if x == 1 else [0, 1, 0]
        train_label = np.asarray([t(x) for x in train_dataset[ft[-1]]])
        print(train_dataset["CanonicalSmiles"].isna().sum())

        train_dataset = MolData(train_dataset["CanonicalSmiles"].values, train_label, train_dataset[ft[1]].values)

        test_label = np.asarray([t(x) for x in test_dataset[ft[-1]]])
        test_dataset = MolData(test_dataset["CanonicalSmiles"].values, test_label, test_dataset[ft[1]].values)

        pickle.dump(train_dataset, open(train_path+".p", "wb"))
        pickle.dump(test_dataset, open(test_path+".p", "wb"))

    loss_fun = torch.nn.CrossEntropyLoss()
    if mode == 'cv':
        # Run k-fold cross-validation
        s = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=random_seed)

        perm = list(itertools.product(*[model_params, optim_grid]))

        lp = len(perm)
        pool = ProcessingPool(nodes=args.num_process)
        k_res = pool.map(thread_fun, perm, [loss_fun]*lp, [args.batch]*lp, [args.e]*lp, 
                         [copy.deepcopy(s)]*lp, list(range(lp)), list(range(lp)), 
                         [copy.deepcopy(train_dataset)]*lp, [device]*lp)
        pool.join()

        k_res.sort(key=lambda x: x["val_avg_metr"][list(x["val_avg_metr"].keys())[0]], reverse=True)
        pickle.dump(k_res, open(args.exp_name + "_results.p", "wb"))

        with open(args.exp_name + "_results.txt", "w") as f:
            for rank, top in enumerate(k_res[:5]):
                f.write("***** Rank:" + str(rank + 1) + " *****")
                f.flush()
                f.write(top["model"].__repr__() + "\n")
                f.flush()
                f.write(top["optimizer"].__repr__() + "\n")
                f.flush()
                for key in list(top.keys())[:-2]:
                    f.write(key + " = " + str(top[key]) + "\n")
                    f.flush()
                f.write("\n\n")
                f.flush()
            f.close()

    else:
        # Risk Assessment
        # TODO: change these parameters according to the best configuration selected
        params = [[129, 64, 3], 4]
        net = NeFPNN(params, device).to(device)
        optim = {'name': 'Adam', 'lr': 5e-05}
        
        h, best, b = net.fit(train_dataset, test_dataset, metrics=metrics, metric_scorer=metric_scorer,
                             batch=args.batch, epochs=args.e, path='refit', criterion=loss_fun,
                             optimizer=optim)
        draw_comparison(h["tr_loss"], h["ts_loss"], ["TR", "TS"], b["epoch"], name="loss_plot.png")
        draw_comparison_metrics(h["tr_metrics"], h["ts_metrics"], ["TR", "TS"], b["epoch"], name="metrics_plot.png")
        with open(args.exp_name + "_results.txt", "w") as f:
            f.write(str(best))
            f.flush()
            f.close()
        t = time.time() - start

