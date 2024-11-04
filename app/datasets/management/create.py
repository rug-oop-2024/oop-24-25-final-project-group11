import streamlit as st
import pandas as pd
from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset


def main():
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
                automl.registry.register(dataset)  # Assuming a register method exists
                st.success(f"Dataset '{uploaded_file.name}' uploaded successfully!")

            except Exception as e:
                st.error(f"Failed to create and save dataset: {e}")
