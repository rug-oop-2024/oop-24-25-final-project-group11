from autoop.core.ml.model import Model
import numpy as np
from sklearn.linear_model import Ridge as SKRidge


class Ridge(Model):
    """Ridge Regression model for regression tasks."""

    def __init__(self, alpha: float = 1.0) -> None:
        """Initialize the model with optional regularization parameter (alpha).

        Args:
            alpha (float): Regularization strength. Default is 1.0.
        """
        super().__init__(name="Ridge")
        self.type = "regression"
        self._model = SKRidge(alpha=alpha)  # Ridge regression with specified alpha

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model with training data.

        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Training targets.
        """
        self._parameters = {
            param: value
            for param, value in self._model.get_params().items()
            if param in ("alpha", "coef_", "intercept_")
        }
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the data using trained model.

        Args:
            X (np.ndarray): Features for prediction.
        """
        return self._model.predict(X)
