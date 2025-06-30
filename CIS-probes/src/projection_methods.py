import numpy as np
from typing import List, Dict, Any
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import LabelEncoder

def run_linear_classifier(X: np.ndarray, y: np.ndarray) -> dict:
    clf = LogisticRegression(max_iter=1000)
    scores = cross_val_score(clf, X, y, cv=5)
    preds = cross_val_predict(clf, X, y, cv=5)
    return {'score': scores.mean(), 'preds': preds}

def run_linear_regression(X: np.ndarray, y: np.ndarray) -> dict:
    reg = LinearRegression()
    scores = cross_val_score(reg, X, y, cv=5, scoring='r2')
    preds = cross_val_predict(reg, X, y, cv=5)
    return {'score': scores.mean(), 'preds': preds}

def run_mlp_classifier(X: np.ndarray, y: np.ndarray) -> dict:
    clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=1000)
    scores = cross_val_score(clf, X, y, cv=5)
    preds = cross_val_predict(clf, X, y, cv=5)
    return {'score': scores.mean(), 'preds': preds}

def run_mlp_regressor(X: np.ndarray, y: np.ndarray) -> dict:
    reg = MLPRegressor(hidden_layer_sizes=(64,), max_iter=1000)
    scores = cross_val_score(reg, X, y, cv=5, scoring='r2')
    preds = cross_val_predict(reg, X, y, cv=5)
    return {'score': scores.mean(), 'preds': preds}

def project_column(X: np.ndarray, y: np.ndarray, task_type: str = 'auto') -> Dict[str, Any]:
    results = {}
    if task_type == 'auto':
        if len(np.unique(y)) < 10 and y.dtype.kind in {'i', 'O', 'U', 'S'}:
            task_type = 'classification'
        else:
            task_type = 'regression'
    if task_type == 'classification':
        results['linear_classifier'] = run_linear_classifier(X, y)
        results['mlp_classifier'] = run_mlp_classifier(X, y)
    else:
        results['linear_regression'] = run_linear_regression(X, y)
        results['mlp_regressor'] = run_mlp_regressor(X, y)
    return results

def encode_labels(y: List[Any]) -> np.ndarray:
    le = LabelEncoder()
    return le.fit_transform(y) 