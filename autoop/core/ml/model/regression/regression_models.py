from autoop.core.ml.model import Model
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.svm import SVC


class Lasso(Model):
    """Lasso model for regression tasks.

    Args:
        _model: The model object.
        _parameters: A dictionary to save the parameters.
    """
    def __init__(self) -> None:
        super().__init__(name="Lasso")
        self._model = Lasso()
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


class SVM(Model):
    """SVM model for regression tasks.

    Args:
        _model: The model object.
    """
    def __init__(self) -> None:
        super().__init__(name="SVM")
        self._model = SVC()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
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
        self._model.predict(X)
