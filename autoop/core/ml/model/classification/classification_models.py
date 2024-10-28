from collections import Counter
from autoop.core.ml.model import Model
import numpy as np
from pydantic import field_validator
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


class KNN(Model):
    """K Nearest Neighbours for classification tasks.

    Args:
        k (int): The number of neighbours.
    """
    def __init__(self, k: int=3) -> None:
        super().__init__(name="KNearestNeighbours")
        self._k = k

    @field_validator("k")
    def k_greater_than_zero(cls, value):
        if value <= 0:
            raise ValueError("k must be greater than 0.")

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model with training data.

        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Training targets.
        """
        self._parameters = {
            "observations": X,
            "ground_truth": y,
            "k": self._k
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

    def _predict_single(self, observation: np.ndarray) -> np.ndarray:
        """Predict the single data.

        Args:
            observation (np.ndarray): Features for prediction.

        Returns:
            np.ndarray: The predicted target.
        """
        distances = np.linalg.norm(self._parameters["observations"] - observation, axis=1)
        k_indices = np.argsort(distances)[:self._k]
        k_nearest_labels = [self._parameters["ground_truth"][i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]


class RandomForestRegressorModel(Model):
    """Random Forest for classification tasks.

    Args:
        _model: The model object.
    """
    def __init__(self):
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


class SVRModel(Model):
    """SVM for classification tasks.

    Args:
        _model: The model object.
    """
    def __init__(self, *args, **kwargs):
        model = SVR(*args, **kwargs)
        super().__init__(name="SVR", model=model)

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
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the data using trained model.

        Args:
            X (np.ndarray): Features for prediction.
        """
        return self.model.predict(X)