import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch

from data import FingerprintDataset, CompoundDataset
from torch_geometric.data import DataLoader
from train.exp_MoRF import run_MoRF
import train.utils as utils
from models import *
import argparse
import dill
import time

random_seed = 42
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)

def run(args):
    adaptive = (args.model_name != 'MoNN' and args.model_name != 'MoRF')

    # Load train and test datasets 
    if adaptive:
        # Dataset for Adaptive fingerprint
        # It contains the atom/bond features used in https://github.com/XuhanLiu/NGFP + dose
        dataset = CompoundDataset([args.bioval_data_path + '_dataset.csv',
                                   args.bioval_data_path + '_atom.csv',
                                   args.bioval_data_path + '_bond.csv'], bidir=True)
    else:
        # Dataset for static fingerprint
        # It contains precomputed Extended-Connectivity Fingerprint + dose        
        dataset = FingerprintDataset(args.bioval_data_path)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # TODO: change these parameters according to the best configuration selected
    inp_dim = dataset[0].x.size()[1]  # len(train_dataset[0].x[0])  for MoNN
    dose_ft_dim = dataset[0].graph_features.size()[1]
    embs = [inp_dim, 256, 128]
    model = GaNN(mlp_input_size = embs[-1] + dose_ft_dim,
                 hidden_size = [64, 32, 3],
                 conv_size = embs)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)

    model.eval()
    with torch.no_grad():
        data_loader = DataLoader(dataset, batch_size=dataset.__len__())
        data = data_loader.__iter__().__next__()

        x = data.to(device)

        pred = model.forward(x)
        dill.dump(pred, open(args.bioval_res_path, "wb"))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', help='Name of the model to run', type=str, default='GaNN', choices=utils.conf_test)
    parser.add_argument('--model_path', help='The path of the model checkpoint used for biological validation', type=str, required=True) # example: ./final_RF.p
    parser.add_argument('--bioval_data_path', help='The path of the dataset used for biological validation', type=str, required=True) # example: ./sweatlead_FTT-R5-512.csv or ./sweetlead in the case of DGN
    parser.add_argument('--bioval_res_path', help='The saving path of the predictions used for biological validation', type=str, default='./bioval_res.p')
    parser.add_argument('--exp_name', help='The name of the experiment', type=str, default="biological_validation")
    args = parser.parse_args()
    
    beginning = time.time()

    if args.model_name == 'MoRF':
        run_MoRF('bioval', random_seed, args)
    else:
        run(args)
    
    t = time.time() - beginning
    print('Required time for main.py in ' + args.exp_name + ' mode: %d hours, %d minutes, %d seconds, %.3d ms' % (t // 3600, t % 3600 // 60, t % 60, (t % 1) * 1000))