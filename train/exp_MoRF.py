def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix, make_scorer
from sklearn.ensemble import RandomForestClassifier
from train.sklearn_validation import run_gridsearch
import train.utils as utils
import pandas as pd
import pickle
import time
import os


def get_tr_ts_data(simple_cv, train, test):
    y_tr = train[train.columns[-1]]
    y_ts = test[test.columns[-1]]
    x_ts = test[test.columns[:-1]]

    if simple_cv:
        ### x_1, x_2, ...., dose, y
        x_tr = train[train.columns[:-1]]
        group = None
    else:
        ### x_1, x_2, ...., dose, n_atoms, y
        g = [train.columns[-2], train.columns[-3]]
        group = train[g]
        x_tr = train[train.columns[:-2]]

    return  y_tr, x_tr, y_ts, x_ts, group


def run_MoRF(mode, ran_state, args):
    assert mode in ['test', 'cv', 'bioval']

    metrics = {"roc": make_scorer(utils.multiclass_roc_auc_score),
               "avg_per_class_accuracy": make_scorer(utils.per_class_accuracy),
               "f1": make_scorer(f1_score, average="macro"),
               "recall": make_scorer(recall_score, average="macro"),
               "precision": make_scorer(precision_score, average="macro"),
               "accuracy": make_scorer(accuracy_score)}

    if "bioval" in mode:
       rf = pickle.load(open(args.model_path, "rb")) # eg, final_RF.p
       s = pd.read_csv(args.bioval_data_path)          # eg, ./sweatlead_FTT-R5-512.csv
       res = rf.predict(s)
       pickle.dump(open(args.bioval_res_path, "wb"))   # eg, bioval_res.p

    elif "test" in mode:
        if args.simple_cv:
            # Best conf for simple cv
            rf = RandomForestClassifier(class_weight='balanced', criterion='entropy', max_depth= 50,
                                        min_samples_leaf=1, min_samples_split=6, n_estimators=700, random_state=ran_state)
            train = pd.read_csv("./Datasets/train_random_forest-FTT-L1024-R5.csv")
            test = pd.read_csv("./Datasets/test_random_forest-FTT-L1024-R5.csv")
        else:
            # Best conf for complex cv
            rf = RandomForestClassifier(class_weight='balanced', criterion='entropy', max_depth= 40,
                                        min_samples_leaf=1, min_samples_split=30, n_estimators=200, random_state=ran_state)
            train = pd.read_csv("./Datasets/train_random_forest_group-FTT-L512-R5.csv")
            test = pd.read_csv("./Datasets/test_random_forest_group-FTT-L512-R5.csv")
        
        y_tr, x_tr, y_ts, x_ts, group = get_tr_ts_data(args.simple_cv, train, test)

        rf.fit(x_tr, y_tr)
        pickle.dump(rf,open(args.exp_name + "_final_RF.p", "wb"))

        with open(args.exp_name + "_res.txt","w") as f: 
            for k in ['TR', 'TS']:
                f.write(f'{k}\n')
                f.flush()
                x = x_ts if k == 'TS' else x_tr
                y = y_ts if k == 'TS' else y_tr
                for k in metrics.keys():
                    f.write(k + ": " + str(metrics[k](rf, x, y))+"\n")
                    f.flush()
                f.write("conf_mat: "+ str(confusion_matrix(y, rf.predict(x)))+"\n")
                f.flush()
            f.close()

    else:
        for radius in [3,4,5]:
            for length in [128, 512, 1024, 2048, 4096]:
                start = time.time()
                ft = {
                    "c": False, # useChirality 
                    "b": True,  # useBondTypes
                    "f": True,  # useFeatures
                    "l": length,
                    "r": radius
                }

                train_name = './Datasets/static_fp' + ('' if args.simple_cv else '_complex_cv') + '/train_random_forest'
                train_name +=  ("_group-" if not args.simple_cv else "-")\
                               + ("T" if ft["c"] else "F")\
                               + ("T" if ft["b"] else "F")\
                               + ("T" if ft["f"] else "F")\
                               + '-L' + str(ft["l"]) + '-R' + str(ft["r"]) + ".csv"
                test_name = train_name.replace('train', 'test')      
    
                if os.path.exists(train_name) and os.path.exists(test_name):
                    # Read datasets
                    train = pd.read_csv(train_name)
                    test = pd.read_csv(test_name)

                    y_tr, x_tr, y_ts, x_ts, group = get_tr_ts_data(args.simple_cv, train, test)
                else:
                    raise ValueError("Path of the dataset (" + train_name + ") does not exist")
                

                param_grid_rf = {"n_estimators": [30]+[100*i for i in range(1,8)],
                                "criterion": ["gini", "entropy"],
                                "max_depth": [None, 3, 8, 13, 20, 25, 30, 40, 50],
                                "min_samples_split": [2, 10, 20, 30, 40],
                                "min_samples_leaf": [1, 5, 10],
                                "class_weight": ["balanced"]}

                rf = RandomForestClassifier(random_state=ran_state)

                # Run k-fold cross-validation
                cv = args.k if args.simple_cv else {"k":3, "group":group}
                search, res = run_gridsearch(x_tr, y_tr, rf, param_grid_rf, metrics, cv=cv, n_jobs=args.num_process, refit="roc", ran_state=ran_state)
                
                out_name = args.exp_name + ("_group-" if not args.simple_cv else "-")\
                                         + ("T" if ft["c"] else "F")\
                                         + ("T" if ft["b"] else "F")\
                                         + ("T" if ft["f"] else "F")\
                                         + '-L' + str(ft["l"]) + '-R' + str(ft["r"])
                pickle.dump(res, open(out_name + "_RF.p", "wb"))
                with open(out_name + "_RF.txt", "w") as f:
                    f.write(str(mode))
                    f.flush()
                    f.write(str(param_grid_rf))
                    f.flush()
                    if "nested" in mode:
                        f.write(str(res["str"]))
                        f.flush()
                        f.write("\n")
                        f.flush()
                        f.write(str(res))
                        f.flush()
                    elif "cv" in mode:
                        f.write("Best model train confusion matrix: " + str(confusion_matrix(y_tr, search.best_estimator_.predict(x_tr))) + "\n")
                        f.flush()
                        for k in res.keys():
                            f.write(k + ':' + str(res[k][:10]))
                            f.flush()
                            f.write("\n")
                            f.flush()

                    f.close()
                t = time.time() - start
                print('Required time for ' + out_name + ' config: %d hours, %d minutes, %d seconds, %.3d ms' % (t // 3600, t % 3600 // 60, t % 60, (t % 1) * 1000))
