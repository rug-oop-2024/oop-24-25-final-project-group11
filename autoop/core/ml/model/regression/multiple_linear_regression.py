from autoop.core.ml.model import Model
import numpy as np


class MultipleLinearRegression(Model):
    """Multiple Linear Regression model for regression tasks."""
    def __init__(self, asset_path: str = "save/models", data: bytes = b"", version: str = "1.0.0") -> None:
        super().__init__(name="MultipleLinearRegression", asset_path=asset_path, data=data, version=version)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model with training data.

        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Training targets.
        """
        # Add a column of 1 for the intercept
        observations_bias = np.insert(X, 0, 1, axis=1)

        # w= (X^T * X)^-1 * X^T * y
        X_T_X = np.dot(observations_bias.T, observations_bias)
        X_T_X_inv = np.linalg.pinv(X_T_X)
        X_T_y = np.dot(observations_bias.T, y)
        w = np.dot(X_T_X_inv, X_T_y)

        # Store the weights
        self._parameters["weights"] = w

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the data using trained model.

        Args:
            X (np.ndarray): Features for prediction.

        Returns:
            np.ndarray: The predicted targets.
        """

        # Check if the model has been trained
        if self._parameters["weights"] is None:
            raise ValueError("The model is not trained yet.")

        # Add a column of 1 for the intercept
        observations_bias = np.insert(X, 0, 1, axis=1)

        # Read the weights
        beta = self._parameters["weights"]

        # y = X * w
        predictions = np.dot(observations_bias, beta)

        return predictions

    def save(self, file_path: str) -> None:
        """Save the model to a file.

        Args:
            file_path (str): The path where to save the model.
        """
        with open(file_path, "w") as f:
            np.save(f, self._parameters)

    def load(self, file_path: str) -> None:
        """Load the model from the file.

        Args:
            file_path (str): The path where to load the model.
        """
        with open(file_path, "r") as f:
            self._parameters = np.load(f, allow_pickle=True).item()

        if "weights" not in self._parameters:
            raise ValueError("Invalid file. The weights could not be loaded.")
