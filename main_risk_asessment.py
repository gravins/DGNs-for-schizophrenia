import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch

from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score, accuracy_score
from train.draw import draw_comparison, draw_comparison_metrics
from data import FingerprintDataset, CompoundDataset
from torch_geometric.data import DataLoader
from train.train_assess import train
from train.exp_MoRF import run_MoRF
import train.utils as utils
from models import *
import argparse
import warnings
import dill
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

    # TODO: change these parameters according to the best configuration selected
    inp_dim = train_dataset[0].x.size()[1]  # len(train_dataset[0].x[0])  for MoNN
    dose_ft_dim = train_dataset[0].graph_features.size()[1]
    embs = [inp_dim, 256, 128]
    model = GaNN(mlp_input_size = embs[-1] + dose_ft_dim,
                 hidden_size = [64, 32, 3],
                 conv_size = embs).to(device)
    
    optim = {'name': 'Adamax', 'lr': 0.005}

    loss_fun = torch.nn.CrossEntropyLoss()

    t = time.time() - start
    print('\t %d hours, %d minutes, %d seconds, %.3d ms' % (t // 3600, t % 3600 // 60, t % 60, (t % 1) * 1000))

    metrics = [utils.multiclass_roc_auc_score, utils.per_class_accuracy, f1_score, recall_score,
               precision_score, accuracy_score, confusion_matrix]

    if args.num_process > 1 and torch.cuda.is_available():
        args.num_process = 1
        warnings.warn("num_process was decreased to 1 because possible errors with cuda", RuntimeWarning)
    print(model.__repr__())

    #train_dataset = train_dataset.shuffle()
    metrics = [utils.multiclass_roc_auc_score, utils.per_class_accuracy, f1_score, recall_score,
               precision_score, accuracy_score, confusion_matrix]

    # Run training
    b, h = train(train_dataset, model, optim, loss_fun, name=args.exp_name,
                 metrics=metrics, ts_set=test_dataset, batch_size=args.batch, 
                 epochs=args.e, device=device)

    draw_comparison(h["tr_loss"], h["ts_loss"], ["TR", "VAL"], b["epoch"],
                    name="loss_plot.png")

    draw_comparison_metrics(h["tr_metrics"], h["ts_metrics"], ["TR", "VAL"], b["epoch"],
                            name="metrics_plot.png")

    dill.dump({"b":b["epoch"], "h":h}, open(args.exp_name + "-hisotry.p", "wb"))
    
    # Save parameters of the best epoch
    mm = b["best"] 
    torch.save(mm.state_dict(), args.exp_name + '-bestmodel.pt')

    mm.eval()
    with torch.no_grad():
        data_loader = DataLoader(test_dataset, batch_size=test_dataset.__len__())
        data = data_loader.__iter__().__next__()

        x = data.to(device)

        pred = mm.forward(x)
        dill.dump(pred, open(args.exp_name + "_testset_predictions.p", "wb"))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', help='Name of the model to run', type=str, default='GaNN', choices=utils.conf_test)
    parser.add_argument('--e', help='The number of epochs', type=int, default=1500)
    parser.add_argument('--batch', help='The batch size', type=int, default=512)
    parser.add_argument('--simple_cv', help='The type of cross-validation used. If True the simple cross-validation was performed, else complex cross-validation.', action='store_true')
    parser.add_argument('--MoNN_tr_path', help='The path of the MoNN training dataset', type=str, default='') # example: ./Datasets/static_fp_complex_cv/train_random_forest_group-FTT-L1024-R3.csv
    parser.add_argument('--exp_name', help='The name of the experiment', type=str, default="risk_assessment_GaNN")
    parser.add_argument('--num_process', help='The number of concurrent jobs', type=int, default=1)
    args = parser.parse_args()
    
    beginning = time.time()

    if args.model_name == 'MoRF':
        run_MoRF('test', random_seed, args)
    else:
        run(args)
    
    t = time.time() - beginning
    print('Required time for main.py in ' + args.exp_name + ' mode: %d hours, %d minutes, %d seconds, %.3d ms' % (t // 3600, t % 3600 // 60, t % 60, (t % 1) * 1000))
