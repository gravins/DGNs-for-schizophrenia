import torch as T
from torch import nn
from torch.utils.data import DataLoader
from .layer import GraphConv, GraphPool, GraphOutput
import numpy as np
from torch import optim
import time


class NeFPNN(nn.Module):
    def __init__(self, param, device):
        super(NeFPNN, self).__init__()

        self.hid_dim = param[0]
        self.conv_size = param[1]
        self.device = device

        self.gnet = nn.ModuleList()

        self.gnet.append(GraphConv(input_dim=43, conv_width=128, device=self.device))
        for _ in range(self.conv_size - 1):
            self.gnet.append(GraphConv(input_dim=134, conv_width=128, device=self.device))

        self.pool = GraphPool(device=self.device)
        self.gop = GraphOutput(input_dim=134, output_dim=128)
        # self.bn = nn.BatchNorm2d(80)
        # self.bnorm = nn.BatchNorm1d(128)

        self.lin = nn.ModuleList()
        for i in range(1, len(self.hid_dim)):
            self.lin.append(nn.Linear(self.hid_dim[i - 1], self.hid_dim[i]))

    def forward(self, atoms, bonds, edges, graph_ft):
        for conv_layer in self.gnet:
            atoms = conv_layer(atoms, bonds, edges)
            # atoms = self.bn(atoms)
            atoms = self.pool(atoms, edges)
        fp = self.gop(atoms, bonds, edges)

        x = T.cat([fp, graph_ft.unsqueeze(1)], dim=1)
        for l in self.lin:
            x = l(x)
            x = T.tanh(x)
        x = T.log_softmax(x, 1)
        return x

    def fit(self, train_set, valid_set, path, metrics, metric_scorer, batch=512, epochs=1000, early_stop=250, optimizer=None, criterion=nn.CrossEntropyLoss()):
        loader_train = DataLoader(train_set, batch_size=batch, shuffle=True, drop_last=True)

        optimizer = optimizer if optimizer is not None else {"name": "Adam", "lr": 1e-3}
        if isinstance(optimizer, dict):
            if optimizer["name"] == "Adam":
                optimizer = optim.Adam(self.parameters(), lr=optimizer["lr"])
            elif optimizer["name"] == "Adamax":
                optimizer = optim.Adamax(self.parameters(), lr=optimizer["lr"])
            elif optimizer["name"] == "SGD":
                optimizer = optim.SGD(self.parameters(), lr=optimizer["lr"])
        else:
            raise ValueError('optimizer must be a dict')

        best = [-np.inf, 0]
        best_result = dict.fromkeys(["model", "epoch"])
        history = {"tr_loss": [],
                   "tr_metrics": [],
                   "ts_loss": [],
                   "ts_metrics": []}

        for epoch in range(epochs):
            self.train()
            t0 = time.time()
            for Ab, Bb, Eb, yb, gb in loader_train:
                Ab, Bb, Eb, yb, Gb = Ab.to(self.device), Bb.to(self.device), Eb.to(self.device), yb.to(self.device), gb.to(self.device)
                optimizer.zero_grad()
                y_ = self.forward(Ab, Bb, Eb, Gb)
                loss = criterion(y_, T.argmax(yb, 1))
                loss.backward()
                optimizer.step()
            eval_tr = self.evaluate(train_set, criterion, metrics=metrics)
            eval_ts = self.evaluate(valid_set, criterion, metrics=metrics)
            history["tr_loss"].append(eval_tr["loss"])
            history["tr_metrics"].append(eval_tr["metrics"])
            history["ts_loss"].append(eval_ts["loss"])
            history["ts_metrics"].append(eval_ts["metrics"])
            if epoch % 100 == 0:
                print('[Epoch:%d/%d] %.1fs' % (epoch, epochs, time.time() - t0))
                print("\tTrain metrics: ", eval_tr["metrics"])
                print("\tTrain Loss: ", eval_tr["loss"])
                print("\tValid metrics: ", eval_ts["metrics"])
                print("\tValid Loss: ", eval_ts["loss"])
                print("********************************")

            if eval_ts["metrics"][metric_scorer] >= best[0]:
                #T.save(self, path + '.pkg')
                #print('[Performance] '+metric_scorer+' is improved from %f to %f, Save model to %s' % (best[0], eval_ts["metrics"][metric_scorer], path + '.pkg'))
                open(path+".txt", "a").close()
                best[0] = eval_ts["metrics"][metric_scorer]
                best[1] = epoch
                best_result["model"] = path + '.pkg'
                best_result["epoch"] = epoch

            if early_stop is not None and epoch - best_result["epoch"] > early_stop: break
        #pickle.dump(history, open("history-"+path+".p", "wb"))
        #pickle.dump(best_result, open("best-result-"+path+".p", "wb"))

        return history, best, best_result

    def evaluate_metrics(self, pred, true_val, metrics):
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

    def evaluate(self, dataset, loss_fun, metrics, output_fun=lambda x: T.argmax(x, 1), device="cpu"):
        """
        Evaluation of the model over the given dataset
        :param dataset:
        :param loss_fun:
        :param metrics: list of metrics or one single metric
        :param output_fun:
        :param device:
        :return: dict with the evaluation for each metric
        """

        self.eval()
        with T.no_grad():
            data_loader = DataLoader(dataset, batch_size=dataset.__len__())
            Ab, Bb, Eb, yb, gb = data_loader.__iter__().__next__()

            Ab, Bb, Eb, out, Gb = Ab.to(self.device), Bb.to(self.device), Eb.to(self.device), yb.to(self.device), gb.to(self.device)
            pred = self.forward(Ab, Bb, Eb, Gb)

            loss = loss_fun(pred, T.argmax(out, 1))

            return {"loss": loss.cpu(), "metrics": self.evaluate_metrics(output_fun(pred).cpu(), output_fun(out).cpu(), metrics)}

