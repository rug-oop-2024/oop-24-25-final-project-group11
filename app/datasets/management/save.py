import streamlit as st
from app.core.system import AutoMLSystem


def main() -> None:
    """
    Main function for the dataset saving page in Streamlit.

    This function initializes the AutoML system instance,
    checks if a dataset artifact is available in the session state,
    and provides a button for saving the dataset artifact to the artifact registry.
    """
    # Initialize AutoMLSystem singleton
    automl = AutoMLSystem.get_instance()

    st.title("Save Dataset Artifact")

    # Check if dataset artifact is available
    if 'dataset_artifact' in st.session_state:
        dataset_artifact = st.session_state['dataset_artifact']

        if st.button("Save Dataset"):
            # Save to artifact registry
            automl.registry.save(dataset_artifact)
            st.success("Dataset saved successfully!")
    else:
        st.warning("No dataset available to save.")
