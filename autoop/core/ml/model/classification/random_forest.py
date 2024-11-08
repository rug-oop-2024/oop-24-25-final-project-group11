from autoop.core.ml.model import Model
import numpy as np
from sklearn.ensemble import RandomForestRegressor


class RandomForestRegressorModel(Model):
    """Random Forest for classification tasks."""
    def __init__(self):
        """Initialize the model.

        Args:
        _model: The model object.
        """
        super().__init__(name="RandomForestRegressor")
        self._model = RandomForestRegressor()

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model with training data.

         Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Training targets.
        """
        self._parameters = {
            param: value
            for param, value in self._model.get_params().items()
            if param in ("coef_", "intercept_")
        }
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the data using trained model.

        Args:
            X (np.ndarray): Features for prediction.
        """
        return self._model.predict(X)
