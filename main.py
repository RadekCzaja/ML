import os
import numpy as np
from imblearn.metrics import geometric_mean_score
from scipy.stats import ttest_rel
from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, recall_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC
from tabulate import tabulate
from tqdm import tqdm
from imblearn.over_sampling import RandomOverSampler
import kbagging
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import pandas as pd

# Random state
rng = 2137

# init klasyfikatorow

random_forest = RandomForestClassifier(n_estimators=50, max_leaf_nodes=16, random_state=rng, n_jobs=-1)
svm = SVC(random_state=rng)
log = LogisticRegression(random_state=rng, solver='lbfgs', max_iter=1000)
voting = VotingClassifier(
    estimators=[('logistics_regression', log), ('random_forest', random_forest), ('support_vector_machine', svm)],
    voting='hard')

# klasyfikatory
clfs = {
    # 'random_forest': random_forest,
    'log': log,
    'svm': svm
}

ros = RandomOverSampler(random_state=rng)


def Test(dataset_name):
    dataset = np.genfromtxt("datasets/" + dataset_name + ".dat", delimiter=", ", skip_header=24)
    X = dataset[:, :-1]
    where_are_NaNs = np.isnan(X)
    X[where_are_NaNs] = 0
    y = dataset[:, -1].astype(int)

    print("Total number of features", X.shape[1])

    # Walidacja krzyżowa
    n_splits = 5
    n_repeats = 10
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=rng)
    acc_scores = np.zeros((len(clfs), n_splits * n_repeats))
    error = np.zeros((len(clfs), n_splits * n_repeats))
    precision = np.zeros((len(clfs), n_splits * n_repeats))
    recall = np.zeros((len(clfs), n_splits * n_repeats))
    f1 = np.zeros((len(clfs), n_splits * n_repeats))
    gmean = np.zeros((len(clfs), n_splits * n_repeats))

    # BAGGING

    for fold_id, (train, test) in tqdm(enumerate(rskf.split(X, y)), total=n_splits * n_repeats):
        for clf_id, clfs_name in enumerate(clfs):
            clf = clone(BaggingClassifier(base_estimator=clfs[clfs_name], n_estimators=20, random_state=rng))
            clf.fit(X[train], y[train])
            y_pred = clf.predict(X[test])
            acc_scores[clf_id, fold_id] = accuracy_score(y[test], y_pred)
            error[clf_id, fold_id] = 1 - accuracy_score(y[test], y_pred)
            precision[clf_id, fold_id] = average_precision_score(y[test], y_pred)
            recall[clf_id, fold_id] = recall_score(y[test], y_pred)
            f1[clf_id, fold_id] = f1_score(y[test], y_pred)
            gmean[clf_id, fold_id] = geometric_mean_score(y[test], y_pred)

    mean = np.mean(gmean, axis=1)
    std = np.std(gmean, axis=1)

    print('\n\n*******')
    print('\n\nBagging')
    for clf_id, clf_name in enumerate(clfs):
        print("%s: mean: %.3f std: (%.2f)" % (clf_name, mean[clf_id], std[clf_id]))
        SaveResults(dataset_name, clf_id, 'Bagging', acc_scores, error, precision, recall, f1, gmean)

    # kbagging

    for fold_id, (train, test) in tqdm(enumerate(rskf.split(X, y)), total=n_splits * n_repeats):
        for clf_id, clfs_name in enumerate(clfs):
            clf = clone(kbagging.BaggingClf(base_estimator=clfs[clfs_name], n_estimators=20, random_state=rng))
            clf.fit(X[train], y[train])
            y_pred = clf.predict(X[test])
            acc_scores[clf_id, fold_id] = accuracy_score(y[test], y_pred)
            error[clf_id, fold_id] = 1 - accuracy_score(y[test], y_pred)
            precision[clf_id, fold_id] = average_precision_score(y[test], y_pred)
            recall[clf_id, fold_id] = recall_score(y[test], y_pred)
            f1[clf_id, fold_id] = f1_score(y[test], y_pred)
            gmean[clf_id, fold_id] = geometric_mean_score(y[test], y_pred)

    mean = np.mean(gmean, axis=1)
    std = np.std(gmean, axis=1)

    print('\n\n*******')
    print('\n\nkbagging')
    for clf_id, clf_name in enumerate(clfs):
        print("%s: mean: %.3f std: (%.2f)" % (clf_name, mean[clf_id], std[clf_id]))
        SaveResults(dataset_name, clf_id, 'kbagging', acc_scores, error, precision, recall, f1, gmean)

    #     ADABOOST
    for fold_id, (train, test) in tqdm(enumerate(rskf.split(X, y)), total=n_splits * n_repeats):
        for clf_id, clfs_name in enumerate(clfs):
            clf = clone(
                AdaBoostClassifier(base_estimator=clfs[clfs_name], n_estimators=20, algorithm='SAMME',
                                   random_state=rng))
            clf.fit(X[train], y[train])
            y_pred = clf.predict(X[test])
            acc_scores[clf_id, fold_id] = accuracy_score(y[test], y_pred)
            error[clf_id, fold_id] = 1 - accuracy_score(y[test], y_pred)
            precision[clf_id, fold_id] = average_precision_score(y[test], y_pred)
            recall[clf_id, fold_id] = recall_score(y[test], y_pred)
            f1[clf_id, fold_id] = f1_score(y[test], y_pred)
            gmean[clf_id, fold_id] = geometric_mean_score(y[test], y_pred)

    mean = np.mean(acc_scores, axis=1)
    std = np.std(acc_scores, axis=1)

    print('\n\n*******')
    print('Adaboost')
    for clf_id, clf_name in enumerate(clfs):
        print("%s: mean: %.3f std: (%.2f)" % (clf_name, mean[clf_id], std[clf_id]))
        SaveResults(dataset_name, clf_id, 'Adaboost', acc_scores, error, precision, recall, f1, gmean)


def SaveResults(dataset_name, clf_id, method, acc_scores, error, precision, recall, f1, gmean):
    if not os.path.exists('results/%s' % method):
        os.makedirs('results/%s' % method)
    if not os.path.exists('results/%s/%s' % (method, dataset_name)):
        os.makedirs('results/%s/%s' % (method, dataset_name))

    data = {
        'accuracy_results': acc_scores[clf_id],
        'error_results': error[clf_id],
        'precision_results': precision[clf_id],
        'recall_results': recall[clf_id],
        'f1_results': f1[clf_id],
        'gmean_results': gmean[clf_id]
    }

    df = pd.DataFrame(data)
    df.to_csv('results/%s/%s' % (method, dataset_name) + str(clf_id) + ".csv")
    print(df)

    np.save('results/%s/%s/%s' % (method, dataset_name, 'accuracy_results' + str(clf_id)), acc_scores)
    np.save('results/%s/%s/%s' % (method, dataset_name, 'error_results' + str(clf_id)), error)
    np.save('results/%s/%s/%s' % (method, dataset_name, 'precision_results' + str(clf_id)), precision)
    np.save('results/%s/%s/%s' % (method, dataset_name, 'recall_results' + str(clf_id)), recall)
    np.save('results/%s/%s/%s' % (method, dataset_name, 'f1_results' + str(clf_id)), f1)
    np.save('results/%s/%s/%s' % (method, dataset_name, 'gmean_results' + str(clf_id)), gmean)


def Statistics(method_name, dataset_name):
    path = 'results/' + method_name + '/' + dataset_name
    clf_names = [
        'log',
        'svm'
    ]

    data = {
        'accuracy_results': np.load(path + '/accuracy_results.npy'),
        'error_results': np.load(path + '/error_results.npy'),
        'precision_results': np.load(path + '/precision_results.npy'),
        'recall_results': np.load(path + '/recall_results.npy'),
        'f1_results': np.load(path + '/f1_results.npy'),
        'gmean_results': np.load(path + '/gmean_results.npy'),
    }

    for id, name in enumerate(data):

        #         scores = np.load(path + '/accuracy_results.npy')
        #         print(path + '/accuracy_results.npy')
        #         print("\nAccuracy scores:\n", scores.shape)
        alfa = .05
        t_statistic = np.zeros((len(clfs), len(clfs)))
        p_value = np.zeros((len(clfs), len(clfs)))

        for i in range(len(clfs)):
            for j in range(len(clfs)):
                t_statistic[i, j], p_value[i, j] = ttest_rel(data[name][i], data[name][j])

                headers = list(clfs.keys())
                names_column = np.expand_dims(np.array(headers), axis=1)

                t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
                t_statistic_table = tabulate(t_statistic_table, headers)
                p_value_table = np.concatenate((names_column, p_value), axis=1)
                p_value_table = tabulate(p_value_table, headers)

                advantage = np.zeros((len(clfs), len(clfs)))
                advantage[t_statistic > 0] = 1
                advantage_table = tabulate(np.concatenate((names_column, advantage), axis=1), headers)

                significance = np.zeros((len(clfs), len(clfs)))
                significance[p_value <= alfa] = 1
                significance_table = tabulate(np.concatenate((names_column, significance), axis=1), headers)

                stat_better = significance * advantage
            stat_better_table = tabulate(np.concatenate((names_column, stat_better), axis=1), headers)
        print()
        print("***** Statistics " + method_name + " " + dataset_name + " " + name + " *****")
        # print("t-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table)
        print("Advantage:\n", advantage_table)
        # print("Statistical significance (alpha = 0.05):\n", significance_table)
        # print("t-statistic:\n", t_statistic, "\n\np-value:\n", p_value)
        # print("Statistically significantly better:\n", stat_better_table)

for file in os.listdir('datasets'):
    filename = os.path.splitext(file)[0]
    print(filename)
    # Test(filename)
    Statistics('Bagging', filename)
    Statistics('kbagging', filename)
    Statistics('Adaboost', filename)
