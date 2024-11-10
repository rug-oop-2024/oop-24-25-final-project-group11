import streamlit as st
import os
import joblib
from app.core.system import AutoMLSystem
from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.model.regression.lasso import Lasso
from autoop.core.ml.model.regression.multiple_linear_regression import MultipleLinearRegression
from autoop.core.ml.model.classification.k_nearest_neighbours import KNN
from autoop.core.ml.model.classification.random_forest import RandomForestRegressorModel
from autoop.core.ml.model.classification.decision_tree import DecisionTree
from autoop.core.ml.model.regression.ridge import Ridge
from autoop.core.ml.metric import get_metric
from autoop.core.ml.feature import Feature


# Page Configuration
st.set_page_config(page_title="Modelling", page_icon="📈")


# Helper function for styled text
def write_helper_text(text: str):
    """Display styled helper text in Streamlit."""
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


# Page Title
st.write("# ⚙ Modelling")
write_helper_text("In this section, you can design a machine learning pipeline to train a model on a dataset.")

# Initialize the AutoML system
automl = AutoMLSystem.get_instance()
artifact_registry = automl.registry

# Load available Datasets
datasets = automl.registry.list(type="dataset")
dataset_names = [dataset.name for dataset in datasets]
selected_dataset_name = st.selectbox("Select a dataset", dataset_names)

# Load data as DataFrame
if selected_dataset_name:
    selected_dataset = next(ds for ds in datasets if ds.name == selected_dataset_name)
    dataset = selected_dataset.read()
    st.write(f"Selected Dataset: {selected_dataset.name}")
    st.dataframe(dataset.head())

    features = dataset.columns
    st.write("Detected Features:")
    st.write(features)

    input_features_names = st.multiselect("Select Input Features", features)
    input_features = [
        Feature(
            name=feature,
            feature_type="categorical" if dataset[feature].dtype == 'object' else "numerical"
        )
        for feature in input_features_names
    ]

    target_feature_name = st.selectbox("Select Target Feature", features)
    if target_feature_name:
        feature_type = "categorical" if dataset[target_feature_name].dtype == 'object' else "numerical"
        target_feature = Feature(name=target_feature_name, feature_type=feature_type)

    if not input_features_names:
        st.error("Please select at least one input feature.")
    if not target_feature_name:
        st.error("Please select a target feature.")

    if input_features_names and target_feature_name:
        task_type = "classification" if dataset[target_feature_name].dtype == 'object' else "regression"
        st.write(f"Detected Task Type: {task_type}")

        # Model Selection Based on Task Type
        if task_type == "classification":
            models = {
                "K-Nearest Neighbors": KNN(),
                "Random Forest": RandomForestRegressorModel(),
                "DecisionTree": DecisionTree()
            }
            available_metrics = ["accuracy", "precision", "recall", "f1_score"]
        else:
            models = {
                "Lasso": Lasso(),
                "Multiple Linear Regression": MultipleLinearRegression(),
                "Ridge": Ridge()
            }
            available_metrics = ["mean_squared_error", "mean_absolute_error", "r_squared"]

        model_names = list(models.keys())
        selected_model_name = st.selectbox("Select a Model", model_names)
        selected_model = models[selected_model_name]

        split_ratio = st.slider("Training/Test Split Ratio", 0.1, 0.9, 0.8)

        # Metric Selection Based on Task Type
        selected_metrics = st.multiselect("Select Metrics", available_metrics)

        st.write("Pipeline Summary:")
        st.json({
            "Task Type": task_type,
            "Selected Model": selected_model_name,
            "Split Ratio": split_ratio,
            "Metrics": selected_metrics
        })

        if st.button("Train Model"):
            metrics = [get_metric(metric_name) for metric_name in selected_metrics]
            pipeline = Pipeline(
                dataset=selected_dataset,
                input_features=input_features,
                target_feature=target_feature,
                model=selected_model,
                split=split_ratio,
                metrics=metrics
            )
            results = pipeline.execute()
            st.write("Training Results:")
            st.write(results)

        pipeline_name = st.text_input("Enter Pipeline Name", value="my_pipeline")
        pipeline_version = st.text_input("Enter Pipeline Version", value="1.0")

        if st.button("Save Pipeline Components") and pipeline_name and pipeline_version:
            if "selected_model" in st.session_state and "selected_dataset" in st.session_state:
                model = st.session_state.selected_model
                dataset = st.session_state.selected_dataset

                # Set up storage paths
                asset_dir = "assets"
                os.makedirs(asset_dir, exist_ok=True)

                # Serialize and save model
                model_path = os.path.join(asset_dir, f"{pipeline_name}_model_v{pipeline_version}.pkl")
                joblib.dump(model, model_path)

                # Serialize and save dataset
                dataset_path = os.path.join(asset_dir, f"{pipeline_name}_dataset_v{pipeline_version}.pkl")
                joblib.dump(dataset, dataset_path)

                # Register model and dataset as artifacts
                try:
                    automl.registry.register(
                        artifact=model,
                        name=f"{pipeline_name}_model",
                        version=pipeline_version,
                        asset_path=model_path
                    )
                    automl.registry.register(
                        artifact=dataset,
                        name=f"{pipeline_name}_dataset",
                        version=pipeline_version,
                        asset_path=dataset_path
                    )
                    st.success(
                        f"Model and Dataset components of pipeline '{pipeline_name}' version {pipeline_version} saved successfully.")
                except Exception as e:
                    st.error(f"Failed to save pipeline components: {str(e)}")
            else:
                st.warning("Please select a model and dataset before saving.")
