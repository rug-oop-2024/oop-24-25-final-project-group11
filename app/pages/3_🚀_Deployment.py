import streamlit as st
import pandas as pd
from app.core.system import AutoMLSystem
from autoop.core.ml.pipeline import Pipeline

# Page Configuration
st.set_page_config(page_title="Deployment", page_icon="🚀")

# Helper function for styled text
def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)

# Page Title
st.write("# 🚀 Deployment")
write_helper_text("In this section, you can deploy a trained pipeline to make predictions on new data.")

# Initialize the AutoML system
automl = AutoMLSystem.get_instance()

# Load Existing Pipelines
pipelines = automl.registry.list(type="pipeline")  # Fetch list of available pipeline names or IDs
pipeline_names = [pipeline.name for pipeline in pipelines]  # Assuming each pipeline has a 'name' attribute
selected_pipeline_name = st.selectbox("Select a Pipeline", pipeline_names)

if selected_pipeline_name:
    # Retrieve and load the pipeline instance
    selected_pipeline = next(p for p in pipelines if p.name == selected_pipeline_name)
    pipeline = automl.registry.get(selected_pipeline_name)  # Retrieve the pipeline artifact
    st.write(f"Selected Pipeline: {pipeline.name} (Version: {pipeline.version})")

    # Display a summary of the pipeline's configuration
    st.json({
        "Task Type": pipeline.task_type,
        "Model": pipeline.model_name,
        "Metrics": pipeline.metrics,
        "Split Ratio": pipeline.split_ratio,
        "Input Features": pipeline.input_features,
        "Target Feature": pipeline.target_feature
    })

    # Upload new data for predictions
    prediction_file = st.file_uploader("Upload CSV file for Predictions", type=["csv"])
    if prediction_file:
        new_data = pd.read_csv(prediction_file)
        st.write("Uploaded Data Preview:")
        st.dataframe(new_data.head())

        # Run Predictions
        if st.button("Run Predictions"):
            predictions = pipeline.predict(new_data)  # Assuming `predict` returns a DataFrame
            st.write("Predictions:")
            st.write(predictions.head())

            # Option to download predictions as CSV
            csv = predictions.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv"
            )
