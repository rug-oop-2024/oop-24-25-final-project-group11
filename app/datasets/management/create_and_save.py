import streamlit as st
import pandas as pd
from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset


def main() -> None:
    """
    Main function for the dataset upload and creation page in Streamlit.

    This function initializes the AutoML system, allows users to upload a CSV file,
    previews the dataset, and then saves it as a Dataset object in the system.
    """
    # Initialize the AutoML system instance
    automl = AutoMLSystem.get_instance()

    # Title of the Streamlit app
    st.title("Upload and Create Dataset")

    # Upload CSV file
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Read the uploaded CSV file into a DataFrame
        df = pd.read_csv(uploaded_file)
        st.write("Preview of the uploaded dataset:")
        st.write(df)

        # Button to confirm upload
        if st.button("Upload Dataset"):
            try:
                # Convert DataFrame to bytes
                dataset_data = df.to_csv(index=False).encode('utf-8')

                # Create the Dataset object
                dataset = Dataset(
                    name=uploaded_file.name,
                    asset_path=f"./datasets/{uploaded_file.name}",
                    data=dataset_data
                )

                # Register the dataset using the AutoML system
                automl.registry.register(dataset)
                st.success(f"Dataset '{uploaded_file.name}' uploaded successfully!")

            except Exception as e:
                st.error(f"Failed to create and save dataset: {e}")
