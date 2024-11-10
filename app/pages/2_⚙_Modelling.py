import streamlit as st
from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.model.regression.lasso import Lasso
from autoop.core.ml.model.regression.multiple_linear_regression import MultipleLinearRegression
from autoop.core.ml.model.classification.k_nearest_neighbours import KNN
from autoop.core.ml.model.classification.random_forest import RandomForestRegressorModel
from autoop.core.ml.model.classification.decision_tree import DecisionTree
from autoop.core.ml.model.regression.ridge import Ridge
from autoop.core.ml.metric import get_metric
from autoop.core.ml.feature import Feature
import numpy as np

# Page Configuration
st.set_page_config(page_title="Modelling", page_icon="📈")


# Helper function for styled text
def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


# Page Title
st.write("# ⚙ Modelling")
write_helper_text("In this section, you can design a machine learning pipeline to train a model on a dataset.")

# Initialize the AutoML system
automl = AutoMLSystem.get_instance()

# Load Datasets
datasets = automl.registry.list(type="dataset")  # Fetch available datasets
dataset_names = [dataset.name for dataset in datasets]
selected_dataset_name = st.selectbox("Select a dataset", dataset_names)

if selected_dataset_name:
    selected_dataset = next(ds for ds in datasets if ds.name == selected_dataset_name)
    dataset = selected_dataset.read()  # Load data as DataFrame
    st.write(f"Selected Dataset: {selected_dataset.name}")
    st.dataframe(dataset.head())

    features = dataset.columns
    st.write("Detected Features:")
    st.write(features)

    input_features_names = st.multiselect("Select Input Features", features)
    input_features = [
        Feature(name=feature, feature_type="categorical" if dataset[feature].dtype == 'object' else "numerical")
        for feature in input_features_names
    ]

    target_feature_name = st.selectbox("Select Target Feature", features)
    if target_feature_name:
        # Determine the feature type based on the dataset column's dtype
        feature_type = "categorical" if dataset[target_feature_name].dtype == 'object' else "numerical"
        # Create a Feature instance
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
            models = {"K-Nearest Neighbors": KNN(),
                      "Random Forest": RandomForestRegressorModel(),
                      "DecisionTree": DecisionTree()
                      }
            available_metrics = [
                "accuracy",
                "precision",
                "recall",
                "f1_score"
            ]
        else:
            models = {
                "Lasso": Lasso(),
                "Multiple Linear Regression": MultipleLinearRegression(),
                "Ridge": Ridge(),
            }
            available_metrics = [
                "mean_squared_error",
                "mean_absolute_error",
                "r_squared"
            ]

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
            # Use get_metric to fetch selected metrics
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
