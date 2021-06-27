"""Model Evaluation Helper Functions

This script is used as a module in the March_Madness_Predictions Jupyter notebooks.

The following functions are present:
    * evaluate_cv_models
    * probs_to_preds
    * test_model_thresholds
    * get_classification_report

Requires a minimum of the 'pandas', 'numpy', and 'sklearn' libraries being present 
in your environment to run.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report


def evaluate_cv_models(cv_models, X, y):
    """Capture stats on model performances against chosen metrics

    Parameters
    ----------
    cv_models : dict
        Collection of model, its parameters, and what CV search to perform
    X : DataFrame
        Historical tournament training dataset
    y : list
        All target variable values

    Returns
    -------
    model_performance : DataFrame
        DataFrame of all models' performance
    """
    # Define CV search parameters and DataFrame to store results
    model_performance = pd.DataFrame(columns=['Mean_Accuracy', 'Mean_Accuracy_Std', 'Mean_AUC', 'Mean_AUC_Std'])
    cross_vals = 4
    scoring = {
        'AUC': 'roc_auc', 
        'Accuracy': 'accuracy',
    }

    for model, params in cv_models.items():
        # Determine which CV search to perform, populate parameters accordingly
        if params[0] == 'Grid':
            model_cv = GridSearchCV(estimator=params[1], param_grid=params[2], cv=cross_vals, scoring=scoring, refit='AUC')
        else:
            model_cv = RandomizedSearchCV(estimator=params[1], param_distributions=params[2], n_iter=100, 
                                        cv=cross_vals, scoring=scoring, refit='AUC', random_state=42)
        # Fit data to model
        model_cv.fit(X, y)
        # Append model itself to cv_models for later use
        cv_models[model].append(model_cv)
        
        # Store model performance with model key in DataFrame
        model_performance.loc[model] = np.round([
            model_cv.cv_results_['mean_test_Accuracy'].mean(),
            model_cv.cv_results_['std_test_Accuracy'].mean(),
            model_cv.cv_results_['mean_test_AUC'].mean(),
            model_cv.cv_results_['std_test_AUC'].mean(),
        ], 3)

    return model_performance


def probs_to_preds(probs, thresh):
    """Convert probabilities to binary target variable predictions

    Parameters
    ----------
    probs : list
        Probabilities of an upset corresponding to each game
    thresh : float
        Threshold for determining whether or not a game is an upset

    Returns
    -------
    list
        Generated target variable predictions
    """
    return [1 if prob > thresh else 0 for prob in probs]


def test_model_thresholds(truths, probs, threshs):
    """Evaluate model performance at various thresholds

    Parameters
    ----------
    truths : list
        Actual target variable values
    probs : list
        Probabilities of an upset corresponding to each game
    threshs : list
        Thresholds against which to test model performance

    Returns
    -------
    performances : DataFrame
        DataFrame used to store prediction performance at all thresholds of interest
    """
    # Define DataFrame
    performances = pd.DataFrame(columns=['Accuracy', 'AUC', 'Upsets (%)'])
    
    for thresh in threshs:
        # Generate predictions from probabilities
        preds = probs_to_preds(probs, thresh)
        
        # Assess performance of predictions
        acc = accuracy_score(truths, preds)
        auc = roc_auc_score(truths, preds)
        pct_upsets = np.mean(preds)

        # Store results
        performances.loc[thresh] = np.round([acc, auc, pct_upsets], 3)
        
    return performances.drop_duplicates(subset=['Accuracy', 'AUC'], keep='last')


def get_classification_report(truths, preds):
    """Generate confusion matrix report based on model performance

    Parameters
    ----------
    truths : list
        Actual target variable values
    preds : list
        Predicted target variable values

    Returns
    -------
    str
        Table with report of interest
    """
    return classification_report(truths, preds)