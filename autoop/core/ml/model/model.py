from abc import abstractmethod, ABC
from autoop.core.ml.artifact import Artifact
import numpy as np
from copy import deepcopy
from typing import Literal, Any
import os
import pickle


class Model(Artifact, ABC):
    """Abstract base class for machine learning models.

    Attributes:
        type: The type of the artifact.
        name: The name of the model.
        asset_path: Path where the model is stored.
        data: Encoded data of the model.
        version: Version of the model.
        _model: The actual model object.
        _parameters: A dictionary to save the parameters of the model.
    """

    def __init__(self, name: str, asset_path: str = "save/models", data: bytes = b"", version: str = "1.0.0") -> None:
        super().__init__(type="model", name=name, asset_path=asset_path, data=data, version=version)
        self._model = None
        self._parameters = {}

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model with training data.

        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Training targets.
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the data using trained model.

        Args:
            X (np.ndarray): Features for prediction.

        Returns:
            np.ndarray: The predicted targets.
        """
        pass

    @property
    def parameters(self):
        """Property access to the model's parameters."""
        return self._parameters

    def save(self, file_path: str) -> None:
        """Save the model to a file.

        Args:
            file_path (str): The path where to save the model.
        """
        if self._model:
            with open(file_path, "w") as f:
                pickle.dump({
                    'model': deepcopy(self._model),
                    'parameters': self._parameters
                }, f)
        else:
            raise ValueError("No model is found.")

    def load(self, file_path: str) -> None:
        """Load the model from the file.

        Args:
            file_path (str): The path where to load the model.
        """
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                data = pickle.load(f)
                self._model = data['model']
                self._parameters = data.get('parameters', {})
        else:
            raise ValueError(f"{file_path} is not exist.")

    def evaluate(self, X: np.ndarray, y_true: np.ndarray, metric: Any) -> float:
        """Evaluate the model using the provided metric.

        Args:
            X (np.ndarray): Features for prediction.
            y_true (np.ndarray): Ground truth targets.
            metric (Any): A metric function to evaluation the predictions.

        Returns:
            float: The calculated metric score.
        """
        y_pred = self.predict(X)
        return metric(y_true, y_pred)
