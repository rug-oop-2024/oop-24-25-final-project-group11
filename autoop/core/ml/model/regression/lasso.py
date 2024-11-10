from autoop.core.ml.model import Model
import numpy as np
from sklearn.linear_model import Lasso as SKlearnLasso


class Lasso(Model):
    """Lasso model for regression tasks."""
    def __init__(self) -> None:
        """Initialize the model.

        Args:
            _model: The model object.
            _parameters: A dictionary to save the parameters.
        """
        super().__init__(name="Lasso")
        self._model = SKlearnLasso()
        self._parameters = {
            "alpha": self._model.alpha,
            "max_iter": self._model.max_iter,
            "tol": self._model.tol,
            "selection": self._model.selection
        }

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight=None) -> None:
        """Train the model with training data.

        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Training targets.
            sample_weight: Sample weight of the training data.
        """
        self._lasso_model.fit(X, y, sample_weight=sample_weight)
        self._parameters["coef_"] = self._model.coef_
        self._parameters["intercept_"] = self._model.intercept_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the data using trained model.

        Args:
            X (np.ndarray): Features for prediction.
        """
        return self._model.predict(X)
