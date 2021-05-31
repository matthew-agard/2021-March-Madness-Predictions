"""Model Selection Helper Functions

This script is used as a module in the March_Madness_Predictions Jupyter notebooks.

The following functions are present:
    * init_knn
    * init_naive_bayes
    * init_logreg
    * init_svm
    * init_rf
    * get_cv_models

Requires a minimum of the 'numpy' and 'sklearn' libraries being installed 
in your environment to run.
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier


def init_knn():
    """Initialize KNN model

    Returns
    -------
    list
        Collection of model, its parameters, and what CV search to perform
    """
    knn = KNeighborsClassifier()
    knn_params = {
        'n_neighbors': np.arange(1, 101),
    }

    return ['Grid', knn, knn_params]


def init_naive_bayes(y):
    """Initialize Naive Bayes model

    Parameters
    -------
    y : list
        Historical tournament game target variables

    Returns
    -------
    list
        Collection of model, its parameters, and what CV search to perform
    """
    gnb = GaussianNB()
    gnb_params = {
        'priors': [None, list(y.value_counts(normalize=True))],
    }

    return ['Grid', gnb, gnb_params]


def init_logreg():
    """Initialize Logistic Regression model

    Returns
    -------
    list
        Collection of model, its parameters, and what CV search to perform
    """
    lr = LogisticRegression()
    lr_params = {
        'C': [10**i for i in range(-5, 6)],
        'max_iter': [500],
        'random_state': [42],
    }

    return ['Grid', lr, lr_params]


def init_svm():
    """Initialize SVM model

    Returns
    -------
    list
        Collection of model, its parameters, and what CV search to perform
    """
    svm = LinearSVC()
    svm_params = {
        'dual': [False],
        'C': [10**i for i in range(-5, 6)],
        'random_state': [42],
    }

    return ['Grid', svm, svm_params]


def init_rf():
    """Initialize SVM model

    Returns
    -------
    list
        Collection of model, its parameters, and what CV search to perform
    """
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
    """Fetch all models

    Parameters
    -------
    y : list
        Historical tournament game target variables

    Returns
    -------
    dict
        Dictionary of all models upon which to perform CV search
    """
    cv_models = {
        'KNN': init_knn(),
        'Naive Bayes': init_naive_bayes(y),
        'LogReg': init_logreg(),
        'SVM': init_svm(),
        'Random Forest': init_rf(),
    }

    return cv_models