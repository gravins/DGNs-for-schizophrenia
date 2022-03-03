from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold, cross_validate
from ._split import *


def nestedCrossValidation(estimator, grid, x, y, inner_split, outer_split, score_metrics, refit=True, shuffle=True, n_jobs=1, n_iter_search=None, ran_state=42):
    """
    Function that perform  nestesd cross validation.
    :param estimator: scikit-learn estimator
    :param grid: [dict] parameter settings to test
    :param x: features
    :param y: targets
    :param inner_split: number of split into inner cross validation
    :param outer_split: number of split into inner cross validation
    :param shuffle: if True shuffle data before splitting
    """

    inner_cv = KFold(n_splits=inner_split, shuffle=shuffle, random_state=ran_state)
    outer_cv = KFold(n_splits=outer_split, shuffle=shuffle, random_state=ran_state)

    if n_iter_search is None:

        clf = GridSearchCV(estimator=estimator, param_grid=grid, cv=inner_cv, scoring=score_metrics, n_jobs=n_jobs, refit=refit)
    else:
        clf = RandomizedSearchCV(estimator=estimator, param_distributions=grid, n_iter=n_iter_search, cv=inner_cv, scoring=score_metrics, n_jobs=n_jobs, refit=refit, random_state=ran_state)

    nested_score = cross_validate(clf, scoring=score_metrics, X=x, y=y, cv=outer_cv, n_jobs=n_jobs)

    r = {"nested_score": nested_score,
         "mean": [(n, nested_score["test_"+n].mean()) for n in score_metrics.keys()],
         "std": [(n, nested_score["test_"+n].std()) for n in score_metrics.keys()],
         "str": [n + str(nested_score["test_"+n].mean()) + "+/-" + str(nested_score["test_"+n].std()) for n in score_metrics.keys()]}
    return r


def report(grid_scores, n_top=3, metrics_names="avg_per_class_accuracy"):
    """
    Report top n_top parameters settings, default n_top=3

    :param grid_scores: output from grid or random search
    :param n_top: how many to report, of top models
    :param metrics_names: name of the metrics to look at
    :return: top_params: [dict] top parameter settings found in
                  search
    """
    if len(grid_scores["mean_test_" + metrics_names[0]]) < n_top:
        n_top = len(grid_scores["mean_test_" + metrics_names[0]])

    top_scores = {"params": []}
    for mn in metrics_names:
        top_scores["mean_test_" + mn] = []
        top_scores["std_test_" + mn] = []

        if "mean_train_" + mn in grid_scores.keys():
            top_scores["mean_train_" + mn] = []
            top_scores["std_train_" + mn] = []

    rank = grid_scores["rank_test_" + metrics_names[0]].tolist()
    i = 1
    while n_top > 0:
        ii = [ind for ind, val in enumerate(rank) if val == i]
        if n_top < len(ii):
            for k in top_scores.keys():
                for ind in ii[:n_top]:
                    top_scores[k].append(grid_scores[k][ind])
            n_top = 0
        else:
            for k in top_scores.keys():
                for ind in ii:
                    top_scores[k].append(grid_scores[k][ind])
            n_top = n_top - len(ii)
        i = (i + len(ii))

    return top_scores


def run_gridsearch(X, y, clf, param_grid, score_metrics, refit=True, cv=5, n_jobs=3, ran_state=42):
    """
    Run a grid search for best estimator parameters.
    :param X: features
    :param y: targets (classes)
    :param clf: scikit-learn classifier
    :param param_grid: [dict] parameter settings to test
    :param score_metrics: scoring metric of dict of scoring metrics
    :param refit: string denoting the scorer that would be used to find the best parameters for refitting the estimator at the end
    :param cv: int or dict, contains number of fold of cross-validation, default 5. 
               In case of a dict is passed, it should be in the form:
                    {
                        "k":number of folds, 
                        "group": array-like, shape (n_samples, n_attributes). Attribute associated to each sample, whose value distribution is preserved in each fold.
                    }
    :param n_jobs: number of jobs to run in parallel
    :return top_params: [dict] from report()
    """
    if isinstance(cv, dict):
        das = DAStratifiedKFold(n_splits=cv["k"], random_state=ran_state, shuffle=True)
        cv = das.split(X, y, cv["group"])

    grid_search = GridSearchCV(clf,
                               scoring=score_metrics,
                               param_grid=param_grid,
                               refit=refit,
                               return_train_score=True,
                               cv=cv, n_jobs=n_jobs)

    grid_search.fit(X, y)
    k = refit if isinstance(refit, str) else "score"
    mn = [k]
    if isinstance(refit, str):
        mn += [n for n in score_metrics.keys() if n != refit]
    return grid_search, report(grid_search.cv_results_, len(grid_search.cv_results_["mean_test_" + k]), mn)


def run_randomsearch(X, y, clf, param_dist, score_metrics, cv=5, refit=True, n_iter_search=20, n_jobs=3, ran_state=42):
    """
    Run a random search for best estimator parameters.
    :param X: features
    :param y: targets (classes)
    :param clf: scikit-learn classifier
    :param param_dist: [dict] distributions of parameters to sample
    :param score_metrics: scoring metric of dict of scoring metrics
    :param refit: string denoting the scorer that would be used to find the best parameters for refitting the estimator at the end
    :param cv: int or dict, contains number of fold of cross-validation, default 5. 
               In case of a dict is passed, it should be in the form:
                    {
                        "k":number of folds, 
                        "group": array-like, shape (n_samples, n_attributes). Attribute associated to each sample, whose value distribution is preserved in each fold.
                    }
    :param n_iter_search: number of random parameter sets to try, default 20
    :param n_jobs: number of jobs to run in parallel
    :return top_params: [dict] from report()
    """
    if isinstance(cv, dict):
        das = DAStratifiedKFold(n_splits=cv["k"], random_state=ran_state, shuffle=True)
        cv = das.split(X, y, cv["group"])

    random_search = RandomizedSearchCV(clf,
                                       scoring=score_metrics,
                                       param_distributions=param_dist,
                                       n_iter=n_iter_search,
                                       cv=cv, n_jobs=n_jobs,
                                       refit=refit,
                                       return_train_score=True,
                                       random_state=ran_state)

    random_search.fit(X, y)
    k = refit if isinstance(refit, str) else "score"
    mn = [k]
    if isinstance(refit, str):
        mn += [n for n in score_metrics.keys() if n != refit]
    return random_search, report(random_search.cv_results_, len(random_search.cv_results_["mean_test_" + k]), mn)
 
