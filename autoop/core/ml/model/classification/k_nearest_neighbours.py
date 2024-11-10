from collections import Counter
from autoop.core.ml.model import Model
import numpy as np
from pydantic import Field, validator

class KNN(Model):
    """K-Nearest Neighbours for classification tasks."""

    k: int = Field(default=3, gt=0)

    def __init__(self, k: int = 3) -> None:
        """Initialize the model with a specified number of neighbors.

        Args:
            k (int): The number of neighbors to use for classification.
        """
        super().__init__(name="KNearestNeighbours")
        self.type = "classification"
        self.k = k

    @validator("k", check_fields=False)
    def k_greater_than_zero(cls, value):
        """Ensure that k is greater than zero."""
        if value <= 0:
            raise ValueError("k must be greater than 0.")
        return value

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model with training data.

        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Training targets.
        """
        self._parameters = {
            "observations": X,
            "ground_truth": y,
            "k": self.k
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the data using the trained model.

        Args:
            X (np.ndarray): Features for predictions.

        Returns:
            np.ndarray: The predicted targets.
        """
        predictions = [self._predict_single(x) for x in X]
        return predictions

    def _predict_single(self, X: np.ndarray) -> str:
        """Predict a single instance.

        Args:
            X (np.ndarray): Features for prediction.

        Returns:
            int: The predicted target.
        """
        distances = np.linalg.norm(self._parameters["observations"] - X, axis=1)
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self._parameters["ground_truth"][i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]
