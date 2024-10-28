from abc import ABC, abstractmethod
from typing import Any
import numpy as np

METRICS = [
    "mean_squared_error",
    "accuracy",
    "mean_absolute_error",
    "r_squared",
    "precision",
    "recall",
    "f1_score",
] # add the names (in strings) of the metrics you implement

def get_metric(name: str):
    # Factory function to get a metric by name.
    # Return a metric instance given its str name.
    if name == "mean_squared_error":
        return MeanSquaredError()
    elif name == "accuracy":
        return Accuracy()
    elif name == "mean_absolute_error":
        return MeanAbsoluteError()
    elif name == "r_squared":
        return RSquared()
    elif name == "precision":
        return Precision()
    elif name == "recall":
        return Recall()
    elif name == "f1_score":
        return F1Score()
    else:
        raise ValueError(f"Unknown metric: {name}")


class Metric(ABC):
    """Base class for all metrics.
    """
    # your code here
    # remember: metrics take ground truth and prediction as input and return a real number

    @abstractmethod
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate the metric between the ground truth (y_true) and the predictions (y_pred).
        Args:
            y_true (np.ndarray): Ground truth values.
            y_pred (np.ndarray): Predicted values.
        Returns:
            float: The computed metric value.
        """
        pass


# add here concrete implementations of the Metric class
class MeanSquaredError(Metric):
    """Mean Squared Error metric for regression tasks."""

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the Mean Squared Error.
        Args:
            y_true (np.ndarray): Ground truth values.
            y_pred (np.ndarray): Predicted values.
        Returns:
            float: Mean Squared Error.
        """
        return np.mean((y_true - y_pred) ** 2)


class Accuracy(Metric):
    """Accuracy metric for classification tasks."""

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the Accuracy.
        Args:
            y_true (np.ndarray): Ground truth values.
            y_pred (np.ndarray): Predicted values.
        Returns:
            float: Accuracy.
        """
        return np.mean(y_true == y_pred)


class MeanAbsoluteError(Metric):
    """Mean Absolute Error metric for regression tasks."""

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the Mean Absolute Error.
        Args:
            y_true (np.ndarray): Ground truth values.
            y_pred (np.ndarray): Predicted values.
        Returns:
            float: Mean Absolute Error.
        """
        return np.mean(np.abs(y_true - y_pred))


class RSquared(Metric):
    """R-squared metric for regression tasks."""

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the R-squared.
        Args:
            y_true (np.ndarray): Ground truth values.
            y_pred (np.ndarray): Predicted values.
        Returns:
            float: R-squared.
        """
        total_variance = np.sum((y_true - np.mean(y_true)) ** 2)
        residual_variance = np.sum((y_true - y_pred) ** 2)
        return 1 - (residual_variance / total_variance)


class Precision(Metric):
    """Precision metric for classification tasks."""

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute precision for binary classification.

        Args:
            y_true (np.ndarray): Ground truth values (binary labels: 0 or 1).
            y_pred (np.ndarray): Predicted values (binary labels: 0 or 1).

        Returns:
            float: Precision (TP / (TP + FP)).
        """
        true_positive = np.sum((y_pred == 1) & (y_true == 1))
        false_positive = np.sum((y_pred == 1) & (y_true == 0))
        if true_positive + false_positive == 0:
            return 0.0
        return true_positive / (true_positive + false_positive)


class Recall(Metric):
    """Recall metric for classification tasks."""

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute recall for binary classification.

        Args:
            y_true (np.ndarray): Ground truth values (binary labels: 0 or 1).
            y_pred (np.ndarray): Predicted values (binary labels: 0 or 1).

        Returns:
            float: Recall (TP / (TP + FN)).
        """
        true_positive = np.sum((y_pred == 1) & (y_true == 1))
        false_negative = np.sum((y_pred == 0) & (y_true == 1))
        if true_positive + false_negative == 0:
            return 0.0
        return true_positive / (true_positive + false_negative)


class F1Score(Metric):
    """F1 Score metric for classification tasks."""

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute F1 score for binary classification.

        Args:
            y_true (np.ndarray): Ground truth values (binary labels: 0 or 1).
            y_pred (np.ndarray): Predicted values (binary labels: 0 or 1).

        Returns:
            float: F1 score, the harmonic mean of precision and recall.
        """
        precision = Precision()(y_true, y_pred)
        recall = Recall()(y_true, y_pred)
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)