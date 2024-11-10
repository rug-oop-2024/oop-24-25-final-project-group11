import uuid
from abc import ABC,abstractmethod
from pydantic import BaseModel, Field
import base64


class Artifact(ABC, BaseModel):
    """Abstract class for all artifacts.
    Attributes:
        id: Unique identifier for the artifact.
        name: Name of the artifact.
        asset_path: Path where the artifact is stored.
        data: Encoded data of the artifact.
        version: Version of the artifact, default="1.0.0"
        type: Type of the artifact (e.g., "dataset").
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field()
    asset_path: str = Field()
    data: bytes = Field()
    version: str = Field("1.0.0")
    type: str = Field(default="artifact")

    class Config:
        """
        Configuration class for Pydantic's BaseModel settings in the Artifact class.

        Attributes:
            arbitrary_types_allowed (bool): Allows Pydantic to handle fields of arbitrary types.
        """
        arbitrary_types_allowed = True

    def read(self) -> bytes:
        """Abstract method to be implemented by subclasses for reading the data.
        Returns:
            bytes: The binary data read from the artifact.
        """
        if not self.data:
            raise ValueError("No data found in the artifact.")
        else:
            return self.data

    @abstractmethod
    def save(self, data: bytes) -> None:
        """Abstract method to be implemented by subclasses for saving the data.
        Args:
            data(bytes): The binary data to be saved into the artifact.
        """
        pass

    def encode(self) -> str:
        """Encode the binary data using base64
        Returns:
            str: The base64-encoded string representation of the binary `data`.
        """
        return base64.b64encode(self.data).decode('utf-8')

    def decode(self, encode_data: str) -> bytes:
        """Decode base64 encoded data to bytes.
        Args:
            encode_data(str): The base64-encoded string that represents the binary data.
        Returns:
            bytes: The original binary data.
        """
        return base64.b64decode(encode_data)
