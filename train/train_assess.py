import torch
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from torch_geometric.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from pathos.multiprocessing import ProcessingPool
from .validation_utils import *
from copy import deepcopy
import numpy as np
import itertools
import pickle
from .draw import *
import re
import dill
import copy

def train(dataset, model, optimizer, loss_fun, metrics=[f1_score, accuracy_score, confusion_matrix], epochs=10, batch_size=10, ts_set=None, early_stop=False, device="cpu", output_fun=lambda x: torch.argmax(x, 1), name=""):
    """
    Training phase of the model over the dataset
    :return: best_result, a dict containing the best model and the epoch in which the best model is obtained
             hystory, a dict with the history of the training containing the loss and the metrics evaluations per each epoch
    """

    if isinstance(optimizer, dict):
        if optimizer["name"] == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=optimizer["lr"])
        elif optimizer["name"] == "Adamax":
            optimizer = torch.optim.Adamax(model.parameters(), lr=optimizer["lr"])
        elif optimizer["name"] == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=optimizer["lr"])
    model = model.to(device)

    # Undersampling
    # At each epoch ~1.2k elements per class are sampled
    weights = []
    num_samples = [0, 0, 0]
    for el in dataset:
        if not torch.equal(el.y, torch.tensor([[0, 1, 0]], dtype=torch.float)):
            weights.append(1.)
        else:
            weights.append(0.)
        num_samples[output_fun(el.y)] += 1
    weights = [x if x > 0 else 1. / num_samples[1] for x in weights]
    sampler = WeightedRandomSampler(weights, num_samples[0] + num_samples[2] + max(num_samples[0], num_samples[2]), replacement=False)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    history = {"tr_loss": [],
               "tr_metrics": [],
               "ts_loss": [],
               "ts_metrics": []}

    best_result = {"model": [],
                   "epoch": 0}

    best_score = None
    count = 0
    for e in range(epochs):

        model.train()
        for i_batch, batch in enumerate(train_loader):
            inp = batch
            out = batch.y

            inp = inp.to(device)
            out = out.to(device)

            # Reset gradients from previous step
            model.zero_grad()

            preds = model.forward(inp)

            loss = loss_fun(preds, torch.argmax(out, 1))
            loss.backward()
            optimizer.step()

        # Evaluate train
        eval_tr = evaluate(model, dataset, loss_fun, metrics, output_fun, device)
        if e % 200 == 0:
            print("Epoch ", e, " Train metrics: ", eval_tr["metrics"])
            print("Epoch ", e, " Train Loss: ", eval_tr["loss"])
        history["tr_loss"].append(eval_tr["loss"])
        history["tr_metrics"].append(eval_tr["metrics"])

        if ts_set is not None:
            # Evaluate test
            eval_ts = evaluate(model, ts_set, loss_fun, metrics, output_fun, device)
            if e % 200 == 0:
                print("Epoch ", e, " Test metrics: ", eval_ts["metrics"])
                print("Epoch ", e, " Test Loss: ", eval_ts["loss"])
            history["ts_loss"].append(eval_ts["loss"])
            history["ts_metrics"].append(eval_ts["metrics"])
            ev = eval_ts["metrics"][list(eval_ts["metrics"].keys())[0]]
        else:
            ev = eval_tr["metrics"][list(eval_tr["metrics"].keys())[0]]

        count += 1
        if e == 1 or (best_score is not None and best_score <= ev and e > 1):
            torch.save(model.state_dict(), name + '-bestmodel-epochs.pt')
            with open(name + '-bestmodel-epochs.txt', "w") as savefile:
                savefile.write("\n")
                savefile.flush()
                savefile.close()
            best_result["model"] = name+'-bestmodel-epochs.pt'
            best_result["epoch"] = e
            best_result["best"] = copy.deepcopy(model)
            dill.dump(eval_ts["pred"], open(name + "_predizione-TOP.p", "wb"))
            # best_loss = eval_ts["loss"]
            best_score = ev
            count = 0

        if early_stop and count > 200:
            break

        # if e %200==0:
        #     print("------------------------")

    return best_result, history


def avg_std_eval(eval_list):
    """
    :param eval_list: list of dictionary [{'m_1':v_1, .., 'm_i':v_i}_1, .. , {'m_1':v_1, .., 'm_i':v_i}_n]
    :return: eval_m, a dict with the average score of each metrics,
             std_m, a dict with the std score of each metrics,
    """
    eval_m = dict.fromkeys(eval_list[0].keys())
    std_m = dict.fromkeys(eval_list[0].keys())
    for k in eval_m.keys():
        eval_m[k] = eval_list[0][k]
        std_m[k] = [eval_list[0][k]]

    for d in eval_list[1:]:
        for k in d.keys():
            eval_m[k] += d[k]
            std_m[k].append(d[k])

    for k in eval_m.keys():
        eval_m[k] = eval_m[k] / float(len(eval_list))
        std_m[k] = np.std(std_m[k])
    return eval_m, std_m


def avg_res_validation(results):
    """
    :param results: should be the output of train function
    :return: the average and std score of each metrics
    """
    tmp = [h["ts_loss"][b["epoch"]] for b, h in results]
    avg_loss = np.average(tmp)
    std_loss = np.std(tmp)
    tmp = [h["ts_metrics"][b["epoch"]] for b, h in results]
    avg_metr, std_metr = avg_std_eval(tmp)

    tmp = [h["tr_loss"][b["epoch"]] for b, h in results]
    tr_avg_loss = np.average(tmp)
    tr_std_loss = np.std(tmp)
    tmp = [h["tr_metrics"][b["epoch"]] for b, h in results]
    tr_avg_metr, tr_std_metr = avg_std_eval(tmp)

    return tr_avg_metr, tr_std_metr, tr_avg_loss, tr_std_loss, avg_metr, std_metr, avg_loss, std_loss


def nested_cv(inner, outer, model_grid, optimizer_grid, dataset, loss_fun, metrics=[f1_score, accuracy_score, confusion_matrix], epochs=10, batch_size=10, early_stop=False, device="cpu", output_fun=lambda x: torch.argmax(x, 1), name=None, n_workers=1, random_seed=None):
    """
    Perform the nested cross validation
    """
    io_folds = make_k_fold_split(dataset, outer, random_seed)

    results = []

    for i, (tr_val, ts) in enumerate(io_folds):
        if name is not None:
            name = str(i) + "_inner-k-fold_" + name
        top_5 = cross_validation(inner, model_grid, optimizer_grid, tr_val, loss_fun, metrics, epochs, batch_size, early_stop, device, output_fun, name, refit=False, n_workers=n_workers, draw_plot=False, random_seed=random_seed)
        top = top_5[0]

        results.append(train(tr_val, top["model"], top["optimizer"], loss_fun, metrics, epochs, batch_size, ts, early_stop, device, output_fun))

    tr_avg_metr, tr_std_metr, tr_avg_loss, tr_std_loss, avg_metr, std_metr, avg_loss, std_loss = avg_res_validation(results)

    nested_res = {"avg_loss": avg_loss,
                  "std_loss": std_loss,
                  "avg_metr": avg_metr,
                  "std_metr": std_metr,
                  "tr_avg_loss": tr_avg_loss,
                  "tr_std_loss": tr_std_loss,
                  "tr_avg_metr": tr_avg_metr,
                  "tr_std_metr": tr_std_metr,
                  "single_res": results}

    if name is not None:
        name = re.sub(".*inner-k-fold_", "", name)

        pickle.dump(nested_res, open(name + ".p", "wb"))

        with open(name + ".txt", "w") as f:

            f.write("TR class metrics " + str(tr_avg_metr) + " +/-" + str(tr_std_metr) + "\n")
            f.flush()
            f.write("TR loss " + str(tr_avg_loss) + " +/-" + str(tr_std_loss) + "\n")
            f.flush()
            f.write("TS class metrics " + str(avg_metr) + " +/-" + str(std_metr) + "\n")
            f.flush()
            f.write("TS loss " + str(avg_loss) + " +/-" + str(std_loss) + "\n")
            f.flush()
            f.write("All TS loss " + str([h["ts_loss"][b["epoch"]] for b, h in results]))
            f.flush()
            f.write("All TS class metrics " + str([h["ts_metrics"][b["epoch"]] for b, h in results]))
            f.flush()
            f.write("All TR loss " + str([h["tr_loss"][b["epoch"]] for b, h in results]))
            f.flush()
            f.write("All TR class metrics " + str([h["tr_metrics"][b["epoch"]] for b, h in results]))
            f.flush()
            f.write("\n\n")
            f.flush()
            f.close()

    return nested_res


def run_validation(param, folds, loss_fun, metrics, epochs, batch_size, early_stop, device, output_fun, name):
    results = []
    for i, (f_tr, f_ts) in enumerate(folds):
        best, history = train(f_tr, deepcopy(param[0]), param[1], loss_fun, metrics, epochs, batch_size, f_ts, early_stop, device,
                              output_fun, name+'-fold_'+str(i))
        results.append((best, history))

    tr_avg_metr, tr_std_metr, tr_avg_loss, tr_std_loss, avg_metr, std_metr, avg_loss, std_loss = avg_res_validation(results)

    r= {"val_avg_loss": avg_loss,
            "val_std_loss": std_loss,
            "val_avg_metr": avg_metr,
            "val_std_metr": std_metr,
            "tr_avg_loss": tr_avg_loss,
            "tr_std_loss": tr_std_loss,
            "tr_avg_metr": tr_avg_metr,
            "tr_std_metr": tr_std_metr,
            "single_res": results,
            "model": param[0],
            "optimizer": param[1]}
    pickle.dump(r, open(name + "-folds.p", "wb"))
    return r

def cross_validation(k, model_grid, optimizer_grid, dataset, loss_fun, metrics=[f1_score, accuracy_score, confusion_matrix], epochs=10, batch_size=10, early_stop=False, complex_cv=False, device="cpu", output_fun=lambda x: torch.argmax(x, 1), name=None, refit=True, n_workers=1, draw_plot=False, random_seed=None):
    """
    Perform the f-fold cross validation
    """
    if complex_cv:
        group = [[d.x.size(0), d.graph_features[0][0]] for d in dataset]
        pickle.dump(group, open("group.p", "wb"))
    else:
        group = None
        
    folds = make_k_fold_split(dataset, k, group=group, shuffle=True, random_seed=random_seed)
    # folds = make_k_fold_split(dataset, k, complex_cv, random_seed)
    k_res = []

    if n_workers > 1:
        pool = ProcessingPool(nodes=n_workers)
        length = len(list(itertools.product(*[model_grid, optimizer_grid])))
        k_res = pool.map(run_validation, list(itertools.product(*[model_grid, optimizer_grid])), [folds]*length, [loss_fun]*length, [metrics]*length, [epochs]*length, [batch_size]*length, [early_stop]*length, [device]*length, [output_fun]*length, [name + "-" + str(i) for i in range(length)])
        pool.join()
    else:
        for i, param in enumerate(list(itertools.product(*[model_grid, optimizer_grid]))):
            k_res.append(run_validation(param, folds, loss_fun, metrics, epochs, batch_size, early_stop, device, output_fun, name + "-" + str(i)))
    k_res.sort(key=lambda x: x["val_avg_metr"][list(x["val_avg_metr"].keys())[0]], reverse=True)
    #k_res.sort(key=lambda x: x["val_avg_loss"])

    if name is not None:
        pickle.dump(k_res, open(name + ".p", "wb"))

        with open(name + ".txt", "w") as f:
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

    if refit:
        # Run the best model on the training set considering 20% as validation set
        top = k_res[0]
        name2 = name+'-top-res'
        tr_idx, val_idx = stratify_split(dataset, test_size=0.2, shuffle=True, random_seed=random_seed)
        val = dataset[torch.tensor(val_idx)]
        tr = dataset[torch.tensor(tr_idx)]
        epoch = 500
        b, h = train(tr, top['model'], top['optimizer'], loss_fun, metrics, epoch, batch_size, val, early_stop, device, output_fun, name2)
        pickle.dump(b, open(name + "-best.p", "wb"))
        pickle.dump(h, open(name + "-hisotry.p", "wb"))

    if draw_plot:
        if refit:
            # Draw best model after refit
            draw_comparison(h["tr_loss"], h["ts_loss"], ["TR", "VAL"], b["epoch"], name=name2+"-loss_plot.png")
            draw_comparison_metrics(h["tr_metrics"], h["ts_metrics"], ["TR", "VAL"], b["epoch"], name=name2+"-metrics_plot.png")

        # Draw best model fold 0
        draw_comparison(k_res[0]["single_res"][0][1]["tr_loss"], k_res[0]["single_res"][0][1]["ts_loss"], ["TR", "VAL"], k_res[0]["single_res"][0][0]["epoch"], y_label=loss_fun.__repr__().replace("()", ""), name="best_"+name)
        draw_comparison(k_res[0]["single_res"][0][1]["tr_loss"], k_res[0]["single_res"][0][1]["ts_loss"], ["TR", "VAL"], k_res[0]["single_res"][0][0]["epoch"], y_label=loss_fun.__repr__().replace("()", ""), name="log_best_"+name, log_scale=True)
        draw_comparison_metrics(k_res[0]["single_res"][0][1]["tr_metrics"], k_res[0]["single_res"][0][1]["ts_metrics"], ["TR", "VAL"], k_res[0]["single_res"][0][0]["epoch"], name="best_metrics"+name)
        draw_comparison_metrics(k_res[0]["single_res"][0][1]["tr_metrics"], k_res[0]["single_res"][0][1]["ts_metrics"], ["TR", "VAL"], k_res[0]["single_res"][0][0]["epoch"], name="log_best_metrics"+name, log_scale=True)

    return k_res


def evaluate_metrics(pred, true_val, metrics):
    """
    Compute the metrics over the pred and true_val data
    :param pred: predicted output
    :param true_val: true value
    :param metrics: list of metrics or single metric
    :return: dict with the evaluation for each metric
    """
    eval_m = {}
    if isinstance(metrics, list):
        for m in metrics:

            eval_m[m.__name__] = m(true_val, pred) if "matrix" in m.__name__ or "accuracy" in m.__name__ else m(true_val, pred, average="macro")

    else:
        eval_m[metrics.__name__] = metrics(true_val, pred)

    return eval_m


def evaluate(model, dataset, loss_fun, metrics=[f1_score, accuracy_score, confusion_matrix], output_fun=lambda x: torch.argmax(x, 1), device="cpu"):
    """
    Evaluation of the model over the given dataset
    :param model:
    :param dataset:
    :param loss_fun:
    :param metrics: list of metrics or one single metric
    :param output_fun:
    :param device:
    :return: dict with the evaluation for each metric
    """

    model.eval()
    with torch.no_grad():
        data_loader = DataLoader(dataset, batch_size=dataset.__len__())
        data = data_loader.__iter__().__next__()

        x = data.to(device)
        out = data.y.to(device)

        pred = model.forward(x)
        ################################################################################################################################## TODO: preds = (torch.sigmoid(preds) > 0.5).detach().cpu()
        loss = loss_fun(pred, torch.argmax(out, 1))

        return {"loss": loss.cpu(), "metrics": evaluate_metrics(output_fun(pred).cpu(), output_fun(out).cpu(), metrics), "pred":pred}

