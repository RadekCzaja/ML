import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, VotingClassifier
from sklearn.datasets import make_classification
import random
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.base import clone
from scipy.stats import mode
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

#BASE_MODEL = DecisionTreeClassifier()
#ENSAMBLE_SIZE = 5
rng =2137
ros = RandomOverSampler(random_state=rng)
class BaggingClf(ClassifierMixin, BaseEstimator):

    def __init__(self, base_estimator, n_estimators, random_state ):
        self.clfs_ = None
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X, y):
        self.clfs_ = []
        X, y = ros.fit_resample(X, y)
        for i in range(self.n_estimators):
            clf = clone(self.base_estimator)
            bootstrap = np.random.choice(len(X), size=len(X), replace=True)
            # print(X.shape)
            # print(y.shape)
            clf.fit(X[bootstrap], y[bootstrap])
            self.clfs_.append(clf)
        return self

    def predict(self, X):
        predictions = []
        # print(self.clfs_)
        for clf in self.clfs_:
            predictions.append(clf.predict(X))
        # print(predictions)
        predictions = np.array(predictions)
        return mode(predictions, axis=0)[0].flatten()