from autoop.core.ml.model.model import Model
from autoop.core.ml.model.regression.multiple_linear_regression import MultipleLinearRegression
from autoop.core.ml.model.regression.lasso import Lasso
from autoop.core.ml.model.regression.svm import SVM
from autoop.core.ml.model.classification.k_nearest_neighbours import KNN
from autoop.core.ml.model.classification.random_forest import RandomForestRegressor
from autoop.core.ml.model.classification.svr import SVR


REGRESSION_MODELS = [
    "MultipleLinearRegression",
    "Lasso",
    "SVM"
] # add your models as str here

CLASSIFICATION_MODELS = [
    "KNearestNeighbours",
    "RandomForestRegressor",
    "SVR"
] # add your models as str here


def get_model(model_name: str) -> Model:
    """Factory function to get a model by name.

    Args:
        model_name (str): Name of the model to retrieve.

    Returns:
        Model: An instance of the requested model.
    """
    if model_name in REGRESSION_MODELS:
        if model_name == "MultipleLinearRegression":
            return MultipleLinearRegression()
        elif model_name == "Lasso":
            return Lasso()
        elif model_name =="SVM":
            return SVM()

    elif model_name in CLASSIFICATION_MODELS:
        if model_name == "KNearestNeighbours":
            return KNN()
        elif model_name == "RandomForestRegressor":
            return RandomForestRegressor()
        elif model_name == "SVR":
            return SVR()

    else:
        raise ValueError(f"Model '{model_name}' not recognized. Please check available models.")
