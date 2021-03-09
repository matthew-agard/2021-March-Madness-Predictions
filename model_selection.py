import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier


def init_knn():
    knn = KNeighborsClassifier()
    knn_params = {
        'n_neighbors': np.arange(1, 101),
    }

    return ['Grid', knn, knn_params]


def init_naive_bayes(y):
    gnb = GaussianNB()
    gnb_params = {
        'priors': [None, list(y.value_counts(normalize=True))],
    }

    return ['Grid', gnb, gnb_params]


def init_logreg():
    lr = LogisticRegression()
    lr_params = {
        'C': [10**i for i in range(-5, 6)],
        'max_iter': [500],
        'random_state': [42],
    }

    return ['Grid', lr, lr_params]


def init_svm():
    svm = LinearSVC()
    svm_params = {
        'dual': [False],
        'C': [10**i for i in range(-5, 6)],
        'random_state': [42],
    }

    return ['Grid', svm, svm_params]


def init_rf():
    rf = RandomForestClassifier()
    rf_params = {
        'n_estimators': np.arange(50, 250),
        'criterion': ['gini', 'entropy'],
        'min_samples_split': [2**i for i in range(1, 10)],
        'min_samples_leaf': [2**i for i in range(1, 10)],
        'random_state': [42],
    }

    return ['Random', rf, rf_params]


def get_cv_models(y):
    knn = init_knn()
    gnb = init_naive_bayes(y)
    lr = init_logreg()
    svm = init_svm()
    rf = init_rf()

    cv_models = {
    'KNN': knn,
    'Naive Bayes': gnb,
    'LogReg': lr,
    'SVM': svm,
    'Random Forest': rf,
    }

    return cv_models