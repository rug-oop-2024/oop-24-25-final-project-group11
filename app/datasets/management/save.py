import streamlit as st
from app.core.system import AutoMLSystem

def main():
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
        st.warning("No dataset available to save. Please upload and create a dataset first.")