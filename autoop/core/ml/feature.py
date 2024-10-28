
from pydantic import BaseModel, Field
from typing import Literal
import numpy as np

from autoop.core.ml.dataset import Dataset

class Feature(BaseModel):
    """The `Feature` class represents a feature (or column) within a dataset in a machine learning context.
    Attributes:
        name: The name of the feature.
        feature_type: Type of the feature, either "numerical" or "categorical"
    """
    # attributes here
    name: str = Field()
    feature_type: Literal['numerical', 'categorical'] = Field()

    @property
    def type(self):
        """Alias for feature_type."""
        return self.feature_type

    def __str__(self):
        """Returns a formatted string representation of the feature, including its name and type."""
        raise f"Feature name: {self.name}; Feature type: {self.feature_type}"