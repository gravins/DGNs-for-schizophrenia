from models import *
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score, confusion_matrix
import itertools

conf = {
    'LinNN': {
        'model': LinNN,
        'emb_dims': [[200], [50], [50, 75]],
        'mlp_dims': [[250, 150, 50, 3], [150, 50,10, 3], [100, 50, 10,3], [30, 10, 3], [50, 3], [5, 3], [3]]
    },

    'SAGENN': {
        'model': SAGENN,
        'emb_dims': [[512, 256, 128], [1024, 512], [256, 128], [128, 64],
                     [1024], [512], [256], [128], [64]],
        'mlp_dims': [[512, 128, 3], [256, 128, 3], [128, 64, 3], [64, 32, 3], [32, 16, 3], [128, 3], [64, 3], [3]],
        'batchnorm': [True, False]
    },

    'GaNN': {
        'model': GaNN,
        'emb_dims': [[512, 256, 128], [256, 128], [256, 64], [128, 64], 
                    [1024], [256], [128], [64]],
        'mlp_dims': [[512, 128, 3], [64, 32, 3], [32, 16, 3], [128, 3], [64, 3], [32, 3], [3]],
        'batchnorm': [True, False]
    },

    'ENN': {
        'model': ENN,
        'emb_dims': [[256, 128], [128, 64], [64, 32], [256], [128], [64], [32]],
        'mlp_dims': [[128, 64, 3], [64, 16, 3], [32, 16, 3], [32, 3], [16, 3], [3]],
        'batchnorm': [True, False]
    },

    'NeFPNN': {
        'emb_dims': [4, 3, 2],
        'mlp_dims': [[256, 64, 3], [128, 64, 3], [256, 3], [128, 3], [64, 3], [3]]
    },

    'MoNN':{
        'model': MoNN,
        'mlp_dims': [[512, 256, 128, 64, 32, 16, 8, 3], [512, 128, 32, 8, 3], [512, 128, 32, 3], [512, 128, 3],
                     [256, 64, 3], [512, 64, 3], [128, 64, 3], [512, 3], [256, 3], [64, 3], [3]]
    },

    'MoRF':{}
}

conf_test = ['SAGENN', 'GaNN', 'MoNN', 'MoRF']

def get_LinNN_confs(inp_dim, dose_ft_dim, device):
    models = []
    
    for ed in conf['LiNN']['emb_dims']:
        for mlp_d in conf['LiNN']['mlp_dims']: 
            for agg in ['mean', 'sum']:
                m = conf['LiNN']['model'](mlp_input_size = ed[-1] + dose_ft_dim, 
                                          hidden_size = mlp_d,
                                          embedding_size = [inp_dim] + ed,
                                          agg=agg).to(device)
                models.append(m)
    return models

def get_DGN_confs(model_name, inp_dim, dose_ft_dim, edge_ft_dim=None, device='cpu'):
    models = []
    for ed in conf[model_name]['emb_dims']:
        for mlp_d in conf[model_name]['mlp_dims']:
            if model_name == 'ENN':
                m = conf[model_name]['model'](inp = inp_dim,
                                              num_edge_feat = edge_ft_dim, 
                                              extra_inp = dose_ft_dim, 
                                              conv_size = [inp_dim] + ed,
                                              hidden_size = mlp_d).to(device)
            
            elif model_name == 'NeFPNN':
                m = [[inp_dim+dose_ft_dim] + mlp_d, ed]

            else:
                m = conf[model_name]['model'](mlp_input_size = ed[-1] + dose_ft_dim,
                                              hidden_size = mlp_d,
                                              conv_size = [inp_dim] + ed).to(device)
            models.append(m)
    return models


def get_MoNN_confs(inp_dim):
    models = []
    for mlp_d in conf['MoNN']['mlp_dims']:  
        m = conf['MoNN']['model'](inp_dim, mlp_d)
        models.append(m)

    return models


def get_optims_confs():
    optims = []
    names = ['Adam', 'SGD', 'Adamax']
    lrs = [0.00005, 0.0005, 0.001, 0.005]
    for n, l in itertools.product(names, lrs):
        optims.append({"name": n, "lr": l})
    return optims


def per_class_accuracy(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    single_acc = 0
    for i in range(len(cm)):
        single_acc += (cm[i][i] / float(sum(cm[i])))
    return single_acc / len(cm)


def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)