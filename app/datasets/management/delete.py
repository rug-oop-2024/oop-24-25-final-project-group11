import streamlit as st
from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset


def main() -> None:
    """
    Main function for the dataset deletion page in Streamlit.

    This function initializes the AutoML system, allows users to select a dataset
    from the available ones, and then deletes it from the system.
    """
    # Initialize the AutoML system instance
    automl = AutoMLSystem.get_instance()

    # Title of the Streamlit app
    st.title("Delete Dataset")

    # Fetch the list of available datasets
    datasets = automl.registry.list(type="dataset")

    # Display available datasets
    if datasets:
        st.write("Available Datasets:")
        dataset_names = [dataset.name for dataset in datasets]
        selected_dataset = st.selectbox("Select a dataset to delete", dataset_names)

        # Button to confirm deletion
        if st.button("Delete Dataset", key="delete_button"):
            # try:
                # Find the dataset object based on the selected name
                dataset_to_delete = next((ds for ds in datasets if ds.name.strip() == selected_dataset.strip()), None)
                st.write(dataset_to_delete)

                if dataset_to_delete:
                    # Delete the selected dataset
                    automl.registry.delete(dataset_to_delete.id)
                    st.success(f"Dataset '{selected_dataset}' has been deleted successfully!")
                else:
                    st.error("Dataset not found.")
            # except Exception as e:
            #     st.error(f"Failed to delete dataset: {e}")
    else:
        st.write("No datasets available to delete.")
