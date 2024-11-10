from autoop.core.ml.model import Model
import numpy as np
from sklearn.svm import SVR as SKSVR


class SVR(Model):
    """SVM for classification tasks."""
    def __init__(self, *args, **kwargs):
        """Initialize the model.

        Args:
            _model: The model object.
        """
        self._model = SKSVR(*args, **kwargs)
        super().__init__(name="SVR")

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
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the data using trained model.

        Args:
            X (np.ndarray): Features for prediction.
        """
        return self.model.predict(X)
