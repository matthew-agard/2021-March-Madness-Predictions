import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report


def evaluate_cv_models(cv_models, X, y):
    model_performance = pd.DataFrame(columns=['Mean_Accuracy', 'Mean_Accuracy_Std', 'Mean_AUC', 'Mean_AUC_Std'])
    cross_vals = 4
    scoring = {
        'AUC': 'roc_auc', 
        'Accuracy': 'accuracy',
    }

    for model, params in cv_models.items():
        if params[0] == 'Grid':
            model_cv = GridSearchCV(estimator=params[1], param_grid=params[2], cv=cross_vals, scoring=scoring, refit='AUC')
        else:
            model_cv = RandomizedSearchCV(estimator=params[1], param_distributions=params[2], n_iter=100, 
                                        cv=cross_vals, scoring=scoring, refit='AUC', random_state=42)
        
        model_cv.fit(X, y)
        cv_models[model].append(model_cv)
        
        model_performance.loc[model] = np.round([
            model_cv.cv_results_['mean_test_Accuracy'].mean(),
            model_cv.cv_results_['std_test_Accuracy'].mean(),
            model_cv.cv_results_['mean_test_AUC'].mean(),
            model_cv.cv_results_['std_test_AUC'].mean(),
        ], 3)

    return model_performance


def probs_to_preds(probs, thresh):
    return [1 if prob > thresh else 0 for prob in probs]


def test_model_thresholds(truths, probs, threshs):
    performances = pd.DataFrame(columns=['Accuracy', 'AUC', 'Upsets (%)'])
    
    for thresh in threshs:
        preds = probs_to_preds(probs, thresh)
        
        acc = accuracy_score(truths, preds)
        auc = roc_auc_score(truths, preds)
        pct_upsets = np.mean(preds)
        
        performances.loc[thresh] = np.round([acc, auc, pct_upsets], 3)
        
    return performances.drop_duplicates(subset=['Accuracy', 'AUC'], keep='last')


def get_classification_report(truths, preds):
    return classification_report(truths, preds)