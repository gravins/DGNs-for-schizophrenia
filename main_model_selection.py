import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch

from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score, accuracy_score
from data import FingerprintDataset, CompoundDataset
from train.train_assess import cross_validation
from train.exp_NeFPNN import run_NeFPNN
from train.exp_MoRF import run_MoRF
import train.utils as utils
from models import *
import argparse
import warnings
import time

random_seed = 42
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)


def run(args):
    print("Loading DataSets...")
    start = time.time()
    adaptive = (args.model_name != 'MoNN' and args.model_name != 'MoRF')

    # Load train and test datasets 
    if adaptive:
        # Dataset for Adaptive fingerprint
        # It contains the atom/bond features used in https://github.com/XuhanLiu/NGFP + dose
        train_dataset = CompoundDataset(['./Datasets/NGFP_train_dataset.csv', './Datasets/NGFP_train_atom.csv', './Datasets/NGFP_train_bond.csv'], bidir=True)
        test_dataset = CompoundDataset(['./Datasets/NGFP_test_dataset.csv', './Datasets/NGFP_test_atom.csv', './Datasets/NGFP_test_bond.csv'], bidir=True)
    else:
        # Dataset for static fingerprint
        # It contains precomputed Extended-Connectivity Fingerprint + dose
        
        train_dataset = FingerprintDataset(args.MoNN_tr_path)
        test_dataset = FingerprintDataset(args.MoNN_tr_path.replace('train', 'test'))

    t = time.time() - start
    print('\t %d hours, %d minutes, %d seconds, %.3d ms' % (t // 3600, t % 3600 // 60, t % 60, (t % 1) * 1000))

    print("Building model <model>...")
    start = time.time()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Define optimizers
    optims = utils.get_optims_confs()

    if 'MoNN' in args.model_name:
        inp_dim = len(train_dataset[0].x[0])
        models = utils.get_MoNN_confs(inp_dim)
    else:
        inp_dim = train_dataset[0].x.size()[1]
        dose_ft_dim = train_dataset[0].graph_features.size()[1]
        edge_ft_dim = train_dataset[0].edge_attr.size()[1]
        
        if 'LinNN' in args.model_name:
            models = utils.get_LinNN_confs(inp_dim, dose_ft_dim, device=device)
        else:
            models = utils.get_DGN_confs(args.model_name, inp_dim, dose_ft_dim, edge_ft_dim, device=device)

    loss_fun = torch.nn.CrossEntropyLoss()

    t = time.time() - start
    print('\t %d hours, %d minutes, %d seconds, %.3d ms' % (t // 3600, t % 3600 // 60, t % 60, (t % 1) * 1000))

    metrics = [utils.multiclass_roc_auc_score, utils.per_class_accuracy, f1_score, recall_score,
               precision_score, accuracy_score, confusion_matrix]

    if args.num_process > 1 and torch.cuda.is_available():
        args.num_process = 1
        warnings.warn("num_process was decreased to 1 because possible errors with cuda", RuntimeWarning)
    cross_validation(args.k, models, optims, train_dataset, loss_fun, batch_size=args.batch, 
                     early_stop=True, epochs=args.e, name=args.exp_name, metrics=metrics, complex_cv=args.simple_cv,
                     n_workers=args.num_process, random_seed=random_seed, draw_plot=True, device=device)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', help='Name of the model to run', type=str, default='GaNN', choices=utils.conf.keys())
    parser.add_argument('--e', help='The number of epochs', type=int, default=900)
    parser.add_argument('--batch', help='The batch size', type=int, default=512)
    parser.add_argument('--k', help='The number of inner folds in the k-fold/nested cross-validation', type=int, default=3)
    parser.add_argument('--simple_cv', help='The type of cross-validation. If True the simple cross-validation is performed, else complex cross-validatio is perfomed.', action='store_true')
    parser.add_argument('--MoNN_tr_path', help='The path of the MoNN training dataset', type=str, default='') # example: ./Datasets/static_fp_complex_cv/train_random_forest_group-FTT-L1024-R3.csv
    parser.add_argument('--exp_name', help='The name of the experiment', type=str, default="3-cv-GaNN")
    parser.add_argument('--num_process', help='The number of concurrent jobs', type=int, default=15)
    args = parser.parse_args()
    
    beginning = time.time()

    if args.model_name == 'MoRF':
        run_MoRF('cv', random_seed, args)
    
    elif args.model_name == 'NeFPNN':
        optims = utils.get_optims_confs()
        model_confs = utils.get_DGN_confs('NeFPNN', 128, 1)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        run_NeFPNN('cv', optims, model_confs, random_seed, device, args)
    
    else:
        run(args)
    
    t = time.time() - beginning
    print('Required time for main.py in ' + args.exp_name + ' mode: %d hours, %d minutes, %d seconds, %.3d ms' % (t // 3600, t % 3600 // 60, t % 60, (t % 1) * 1000))
