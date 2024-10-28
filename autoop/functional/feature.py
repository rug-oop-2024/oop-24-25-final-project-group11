import pandas as pd
from typing import List
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature

def detect_feature_types(dataset: Dataset) -> List[Feature]:
    """Assumption: only categorical and numerical features and no NaN values.
    Args:
        dataset: Dataset
    Returns:
        List[Feature]: List of features with their types.
    """
    # Read the dataset to get the DataFrame
    df = dataset.read()
    features = []

    # Iterate every column
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            feature_type = "numerical"
        elif pd.api.types.is_categorical_dtype(df[column]) or df[column].dtype == 'object':
            feature_type = "categorical"
        else:
            raise ValueError(f"Unexpected data type for column '{column}': {df[column].dtype}")

        # Create the feature object
        feature = Feature(name=column, feature_type=feature_type)
        features.append(feature)

    return features
