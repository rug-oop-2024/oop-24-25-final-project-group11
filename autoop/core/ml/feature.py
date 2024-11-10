from pydantic import BaseModel, Field
from typing import Literal


class Feature(BaseModel):
    """
    The `Feature` class represents a feature (or column) within a dataset in a machine learning context.

    Attributes:
        name: The name of the feature.
        feature_type: Type of the feature, either "numerical" or "categorical"
    """
    # attributes here
    name: str = Field()
    feature_type: Literal['numerical', 'categorical'] = Field()

    @property
    def type(self) -> str:
        """
        Alias for feature_type.

        Returns:
            str: The type of the feature, either "numerical" or "categorical".
        """
        return self.feature_type

    def __str__(self) -> str:
        """
        Returns a formatted string representation of the feature, including its name and type.

        Returns:
            str: Formatted string with feature name and type.
        """
        return f"Feature name: {self.name}; Feature type: {self.feature_type}"
