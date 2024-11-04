from autoop.core.ml.artifact import Artifact
import pandas as pd
import io
import base64
from typing import Optional, Dict, List
from pydantic import Field


class Dataset(Artifact):
    """
    Dataset class that extends Artifact to store and manage dataset artifacts.

    Attributes:
        type: The type of the artifact.
        name: The name of the dataset.
        asset_path: The path where the dataset artifact is stored.
        data: Encoded CSV data of the dataset.
        version: Version of the dataset, default is "1.0.0".
        tags: Optional list of tags for metadata purposes.
        metadata: Optional dictionary for additional information.
    """

    tags: Optional[List[str]] = Field(default_factory=list)
    metadata: Optional[Dict[str, str]] = Field(default_factory=dict)

    def __init__(self, *args, **kwargs):
        super().__init__(type="dataset", *args, **kwargs)

    @classmethod
    def from_dataframe(cls, data: pd.DataFrame, name: str, asset_path: str, version: str = "1.0.0",
                       metadata: Optional[Dict[str, str]] = None):
        """
        Factory method to create a Dataset instance from a DataFrame.

        Args:
            data (pd.DataFrame): The DataFrame to convert to a Dataset.
            name (str): The name of the dataset.
            asset_path (str): The path to save the dataset.
            version (str): Version identifier for the dataset.
            metadata (Dict[str, str], optional): Metadata information for the dataset.

        Returns:
            Dataset: An instance of the Dataset class.
        """
        return cls(
            name=name,
            asset_path=asset_path,
            data=data.to_csv(index=False).encode(),
            version=version,
            metadata=metadata or {},
        )

    def read(self) -> pd.DataFrame:
        """
        Reads the dataset artifact and converts it back to a DataFrame.

        Returns:
            pd.DataFrame: The decoded DataFrame from the stored CSV data.
        """
        csv_data = super().read().decode()
        return pd.read_csv(io.StringIO(csv_data))

    def save(self, data: pd.DataFrame) -> None:
        """
        Encodes and stores a DataFrame as CSV data in the dataset artifact.

        Args:
            data (pd.DataFrame): The DataFrame to save.
        """
        self.data = data.to_csv(index=False).encode()
        super().save(self.data)

    def encode(self) -> str:
        """
        Encodes the stored binary data as a base64 string.

        Returns:
            str: Base64-encoded string of the binary data.
        """
        return base64.b64encode(self.data).decode('utf-8') if self.data else ""

    def decode(self, encoded_data: str) -> bytes:
        """
        Decodes a base64-encoded string back to binary data.

        Args:
            encoded_data (str): Base64-encoded string to decode.

        Returns:
            bytes: The original binary data.
        """
        return base64.b64decode(encoded_data)
