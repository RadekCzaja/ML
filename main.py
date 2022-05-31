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

# Random state
rng = 2137

# init klasyfikatorow

random_forest = RandomForestClassifier(n_estimators=50, max_leaf_nodes=16, random_state=rng, n_jobs=-1)
svm = SVC(random_state=rng)
log = LogisticRegression(random_state=rng,solver='lbfgs', max_iter=1000)
voting = VotingClassifier(
    estimators=[('logistics_regression', log), ('random_forest', random_forest), ('support_vector_machine', svm)],
    voting='hard')

# klasyfikatory

clfs = {
   # 'random_forest': random_forest,
    'log': log,
    'svm': svm
    }

def Test(dataset):
    dataset = np.genfromtxt("%s.csv" % (dataset), delimiter=",", skip_header=37)
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

    #BAGGING

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
            print(gmean[clf_id, fold_id])
            print('*****************')
            print(y[test], y_pred)
            print('*****************')
            exit()

    mean = np.mean(gmean, axis=1)
    std = np.std(gmean, axis=1)

    print('\n\nBagging')
    for clf_id, clf_name in enumerate(clfs):
        print("%s: %.3f (%.2f)" % (clf_name, mean[clf_id], std[clf_id]))




    np.save('Bagging/accuracy_results', acc_scores)
    np.save('Bagging/error_results', error)
    np.save('Bagging/precision_results', precision)
    np.save('Bagging/recall_results', recall)
    np.save('Bagging/f1_results', f1)
    np.save('Bagging/gmean_results', gmean)

    #ADABOOST
    #
    # for fold_id, (train, test) in tqdm(enumerate(rskf.split(X, y)), total=n_splits * n_repeats):
    #     for clf_id, clfs_name in enumerate(clfs):
    #         clf = clone(
    #             AdaBoostClassifier(base_estimator=clfs[clfs_name], n_estimators=20, algorithm='SAMME', random_state=rng))
    #         clf.fit(X[train], y[train])
    #         y_pred = clf.predict(X[test])
    #         acc_scores[clf_id, fold_id] = accuracy_score(y[test], y_pred)
    #         error[clf_id, fold_id] = 1 - accuracy_score(y[test], y_pred)
    #         precision[clf_id, fold_id] = average_precision_score(y[test], y_pred)
    #         recall[clf_id, fold_id] = recall_score(y[test], y_pred)
    #         f1[clf_id, fold_id] = f1_score(y[test], y_pred)
    #         gmean[clf_id, fold_id] = geometric_mean_score(y[test], y_pred)
    #
    # mean = np.mean(acc_scores, axis=1)
    # std = np.std(acc_scores, axis=1)
    #
    # print('\n\n*******')
    # print('Adaboost')
    # for clf_id, clf_name in enumerate(clfs):
    #     print("%s: %.3f (%.2f)" % (clf_name, mean[clf_id], std[clf_id]))
    #
    # np.save('Adaboost/accuracy_results', acc_scores)
    # np.save('Adaboost/error_results', error)
    # np.save('Adaboost/precision_results', precision)
    # np.save('Adaboost/recall_results', recall)
    # np.save('Adaboost/f1_results', f1)
    # np.save('Adaboost/gmean_results', gmean)

def Statistics(path):

    scores = np.load(path + '/accuracy_results.npy')
    print(path + '/accuracy_results.npy')
    print("\nAccuracy scores:\n", scores.shape)
    alfa = .05
    t_statistic = np.zeros((len(clfs), len(clfs)))
    p_value = np.zeros((len(clfs), len(clfs)))
    for i in range(len(clfs)):
        for j in range(len(clfs)):
            t_statistic[i, j], p_value[i, j] = ttest_rel(scores[i], scores[j])
    print("t-statistic:\n", t_statistic, "\n\np-value:\n", p_value)

    headers = list(clfs.keys) # TU SĄ BŁEDY
    names_column = np.expand_dims(np.array(list(clfs.keys())), axis=1) # I TU


    t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
    t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
    p_value_table = np.concatenate((names_column, p_value), axis=1)
    p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
    print("t-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table)

    advantage = np.zeros((len(clfs), len(clfs)))
    advantage[t_statistic > 0] = 1
    advantage_table = tabulate(np.concatenate((names_column, advantage), axis=1), headers)
    print("Advantage:\n", advantage_table)

    significance = np.zeros((len(clfs), len(clfs)))
    significance[p_value <= alfa] = 1
    significance_table = tabulate(np.concatenate((names_column, significance), axis=1), headers)
    print("Statistical significance (alpha = 0.05):\n", significance_table)

    stat_better = significance * advantage
    stat_better_table = tabulate(np.concatenate((names_column, stat_better), axis=1), headers)
    print("Statistically significantly better:\n", stat_better_table)

#Is-this-a-good-customer << maly
#Lending-Club-Loan-Data << duzy
#Test('Is-this-a-good-customer')
Statistics('Bagging')
#Statistics('Adaboost')

#keel es
