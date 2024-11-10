import streamlit as st
import pandas as pd
from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset

# Initialize AutoMLSystem singleton
automl = AutoMLSystem.get_instance()

# Page title
st.title("Dataset Management")

# Display available datasets
st.header("Available Datasets")

# Fetch and display datasets from the registry
datasets = automl.registry.list(type="dataset")

if datasets:
    for dataset in datasets:
        st.write(f"- {dataset.name}")

        # Add a button to preview the dataset with editable rows
        with st.expander(f"Preview and Edit {dataset.name}"):
            try:
                # Load the dataset and display a preview
                df = dataset.read()
                st.write("Preview of the dataset:")
                st.write(df)

                # Let the user select a row to edit
                row_to_edit = st.selectbox(
                    f"Select a row to edit in {dataset.name}",
                    df.index,
                    key=f"selectbox_{dataset.name}"
                )

                # Display fields for each column in the selected row
                edited_row = {}
                for col in df.columns:
                    current_value = df.at[row_to_edit, col]

                    # Check the column's data type and provide the appropriate input
                    if pd.api.types.is_numeric_dtype(df[col]):
                        new_value = st.number_input(
                            f"Edit {col}",
                            value=float(current_value),
                            key=f"number_input_{dataset.name}_{col}"
                        )
                    else:
                        new_value = st.text_input(
                            f"Edit {col}",
                            value=str(current_value),
                            key=f"text_input_{dataset.name}_{col}"
                        )

                    # Store the edited value in a dictionary
                    edited_row[col] = new_value

                # Button to save the modifications
                if st.button("Save Changes", key=f"save_button_{dataset.name}"):
                    # Update the DataFrame with the new values
                    for col, value in edited_row.items():
                        df.at[row_to_edit, col] = value

                    # Save the modified data back as a new file (upload as new dataset)
                    try:
                        # Convert the updated DataFrame to CSV bytes
                        dataset_data = df.to_csv(index=False).encode('utf-8')

                        # Create a new Dataset object with updated data
                        new_dataset = Dataset(
                            name=f"modified_{dataset.name}",
                            asset_path=f"./datasets/modified_{dataset.name}",
                            data=dataset_data
                        )

                        # Register the modified dataset in AutoML system
                        automl.registry.register(new_dataset)
                        st.success(f"Dataset updated and saved as '{new_dataset.name}' successfully!")

                    except Exception as e:
                        st.error(f"Failed to save dataset: {e}")

            except Exception as e:
                st.error(f"Could not load dataset: {e}")
else:
    st.write("No datasets available.")

# Expander for creating a new dataset
with st.expander("Upload & Create New Dataset"):
    # File uploader for CSV files
    uploaded_file = st.file_uploader("Choose a CSV file to upload", type="csv")

    if uploaded_file is not None:
        # Read and display preview of the uploaded CSV file
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

                # Register the dataset in AutoML system
                automl.registry.register(dataset)
                st.success(f"Dataset '{uploaded_file.name}' uploaded successfully!")

            except Exception as e:
                st.error(f"Failed to create and save dataset: {e}")
