"""
metrics.py

Evaluation metrics for model performance and experiment results.
Supports classification and regression tasks using scikit-learn.
"""

from typing import List, Optional, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score as sk_accuracy_score,
    precision_score as sk_precision_score,
    recall_score as sk_recall_score,
    f1_score as sk_f1_score,
    mean_squared_error as sk_mean_squared_error,
    mean_absolute_error as sk_mean_absolute_error,
    r2_score as sk_r2_score,
    classification_report as sk_classification_report,
    confusion_matrix as sk_confusion_matrix,
)

def accuracy_score(y_true: Union[List, np.ndarray], y_pred: Union[List, np.ndarray]) -> float:
    """Compute the accuracy classification score."""
    return sk_accuracy_score(y_true, y_pred)

def precision_score(
    y_true: Union[List, np.ndarray], 
    y_pred: Union[List, np.ndarray], 
    average: str = 'binary'
) -> float:
    """Compute the precision score."""
    return sk_precision_score(y_true, y_pred, average=average, zero_division=0)

def recall_score(
    y_true: Union[List, np.ndarray], 
    y_pred: Union[List, np.ndarray], 
    average: str = 'binary'
) -> float:
    """Compute the recall score."""
    return sk_recall_score(y_true, y_pred, average=average, zero_division=0)

def f1_score(
    y_true: Union[List, np.ndarray], 
    y_pred: Union[List, np.ndarray], 
    average: str = 'binary'
) -> float:
    """Compute the F1 score."""
    return sk_f1_score(y_true, y_pred, average=average, zero_division=0)

def mean_squared_error(y_true: Union[List, np.ndarray], y_pred: Union[List, np.ndarray]) -> float:
    """Compute mean squared error for regression."""
    return sk_mean_squared_error(y_true, y_pred)

def mean_absolute_error(y_true: Union[List, np.ndarray], y_pred: Union[List, np.ndarray]) -> float:
    """Compute mean absolute error for regression."""
    return sk_mean_absolute_error(y_true, y_pred)

def r2_score(y_true: Union[List, np.ndarray], y_pred: Union[List, np.ndarray]) -> float:
    """Compute R^2 (coefficient of determination) regression score."""
    return sk_r2_score(y_true, y_pred)

def classification_report(
    y_true: Union[List, np.ndarray], 
    y_pred: Union[List, np.ndarray], 
    target_names: Optional[List[str]] = None
) -> str:
    """Return a text summary of the precision, recall, F1 score for each class."""
    return sk_classification_report(y_true, y_pred, target_names=target_names, zero_division=0)

def confusion_matrix(
    y_true: Union[List, np.ndarray], 
    y_pred: Union[List, np.ndarray]
) -> np.ndarray:
    """Compute confusion matrix to evaluate the accuracy of a classification."""
    return sk_confusion_matrix(y_true, y_pred)

def print_classification_report(
    y_true: Union[List, np.ndarray], 
    y_pred: Union[List, np.ndarray], 
    target_names: Optional[List[str]] = None
):
    """Print classification report and confusion matrix."""
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

def print_regression_report(
    y_true: Union[List, np.ndarray], 
    y_pred: Union[List, np.ndarray]
):
    """Print regression report."""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R^2: {r2:.4f}")

# Example usage:
if __name__ == "__main__":
    # Classification
    y_true = [1, 0, 1, 1, 0, 1]
    y_pred = [1, 0, 0, 1, 0, 1]
    print_classification_report(y_true, y_pred, target_names=["Negative", "Positive"])

    # Regression
    y_true_reg = [2.1, 3.0, 4.5, 6.2]
    y_pred_reg = [2.0, 2.9, 4.4, 6.3]
    print("\nRegression Metrics:")
    print_regression_report(y_true_reg, y_pred_reg)