import numpy as np

def accuracy_score(y_true, y_pred):
    """(TP + TN) / Total"""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return np.mean(y_true == y_pred)

def precision_score(y_true, y_pred):
    """TP / (TP + FP) -> Accuracy of positive predictions"""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fp) if tp + fp > 0 else 0

def recall_score(y_true, y_pred):
    """TP / (TP + FN) -> Fraction of positives correctly identified"""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn) if tp + fn > 0 else 0

def f1_score(y_true, y_pred):
    """2 * (Precision * Recall) / (Precision + Recall)"""
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)   
    return 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
