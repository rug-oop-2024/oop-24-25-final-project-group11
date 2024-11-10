from autoop.core.ml.model import Model
import numpy as np
from sklearn.tree import DecisionTreeClassifier as SKDecisionTree


class DecisionTree(Model):
    """Decision Tree for classification tasks."""

    def __init__(self, *args, **kwargs):
        """Initialize the model.

        Args:
            _model: The model object.
        """
        super().__init__(name="DecisionTree")
        self.type = "classification"
        self._model = SKDecisionTree()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model with training data.

        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Training targets.
        """
        self._parameters = {
            param: value
            for param, value in self._model.get_params().items()
            if param in ("max_depth", "min_samples_split", "min_samples_leaf")
        }
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the data using trained model.

        Args:
            X (np.ndarray): Features for prediction.
        """
        return self._model.predict(X)

    def get_params(self) -> dict:
        """Get the current parameters of the model.

        Returns:
            dict: The model parameters.
        """
        return self._parameters
