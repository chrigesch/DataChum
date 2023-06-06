# Import moduls from local directories
from assets.colors import get_color
from modules.cluster.models import (
    cluster_models_to_evaluate,
    MODELS_WITH_N_CLUSTER,
    MODELS_WITH_N_COMPONENTS,
)
from modules.utils.preprocessing import (
    data_preprocessing,
    _get_feature_names_after_preprocessing,
)

# Import the required libraries
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

# Import libraries for debugging


@st.cache_data(ttl=3600, max_entries=10)
def line_plot(
    data: pd.DataFrame,
    x: str,
    y: str,
    traces: str,
    color: str,
):
    list_colors = get_color(color, len(data[traces].unique()))

    fig_variable = px.line(
        data,
        x=x,
        y=y,
        color=traces,
        color_discrete_sequence=list_colors,
    )

    fig_variable.update_layout(xaxis_type="category", showlegend=True)
    return fig_variable


@st.cache_data(ttl=3600, max_entries=10)
def silhouette_plot(
    X_prep: pd.DataFrame,
    cluster_model: str,
    cluster_labels: np.ndarray,
    color: str,
):
    n_clusters = len(np.unique(cluster_labels))
    # Compute the average silhouette_score
    silhouette_avg = silhouette_score(X_prep, cluster_labels)
    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X_prep, cluster_labels)
    # Initiate list of colors
    list_colors = get_color(color, n_clusters)
    # Create a plot & update title
    fig = go.Figure()
    y_lower = 10
    # Start loop to add traces of every cluster
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        filled_area = go.Scatter(
            y=np.arange(y_lower, y_upper),
            x=ith_cluster_silhouette_values,
            mode="lines",
            showlegend=False,
            line=dict(width=0.5, color=list_colors[i]),
            fill="tozerox",
            text="Cluster " + str(i) + " (n = " + str(size_cluster_i) + ")",
        )
        fig.add_trace(filled_area)
        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    # The vertical line for average silhouette score of all the values
    fig.add_vline(x=silhouette_avg, line_width=1, line_dash="dash", line_color="red")
    fig.update_xaxes(
        range=[0, max(sample_silhouette_values)],
        title="Silhouette coefficient values (avg = "
        + str(round(silhouette_avg, 2))
        + ")",
    )
    fig.update_yaxes(
        title="Cluster label",
        showticklabels=False,
        range=[0, len(X_prep) + (n_clusters + 1) * 10],
    )

    fig.update_layout(
        title=str(cluster_model) + " - Silhouette plot for the various clusters"
    )

    return fig


@st.cache_data(ttl=3600, max_entries=10)
def get_cluster_labels_and_X_prep(
    data: pd.DataFrame,
    imputation_numerical: str,
    imputation_categorical: str,
    scaler: str,
    cluster_model: list,
    n_clusters: int,
):
    # Get categorical and numerical column names
    cols_num = data.select_dtypes(include=["float", "int"]).columns.to_list()
    cols_cat = data.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.to_list()
    pipeline = data_preprocessing(
        cols_num=cols_num,
        cols_cat=cols_cat,
        imputation_numerical=imputation_numerical,
        scaler=scaler,
        imputation_categorical=imputation_categorical,
        one_hot_encoding=True,
    )
    # Data preparation
    data_prep = pipeline.fit_transform(data)
    # Get labels of all features
    labels = _get_feature_names_after_preprocessing(
        pipeline,
        includes_model=False,
    )
    # Convert output to Dataframe and add columns names
    data_prep = pd.DataFrame(data_prep, columns=labels, index=data.index)
    # Get cluster model
    model_list = cluster_models_to_evaluate(models=cluster_model)[0]
    model_name = model_list[0]
    model = model_list[1]
    if model_name in MODELS_WITH_N_COMPONENTS:
        model.set_params(**{"n_components": n_clusters})
    elif model_name in MODELS_WITH_N_CLUSTER:
        model.set_params(**{"n_clusters": n_clusters})
    # Fit the model to get the cluster labels
    cluster_labels = model.fit_predict(data_prep)
    return cluster_labels, data_prep


@st.cache_data(ttl=3600, max_entries=10)
def prepare_results_for_line_plot_metrics(
    data: pd.DataFrame,
    model: str,
):
    # Initiate list with id variables
    cols_id = ["model", "n_clusters"]
    # Get all column names
    cols_all = data.columns.to_list()
    # Filter the selected model
    data = data[data["model"] == model]
    # Compute means of all folds, repetitions or bootstrap samples
    data = data.groupby(by=cols_id).mean().reset_index()
    # Apply MinMax-Scaler to "Calinski-Harabasz" & "Davies-Bouldin"
    if "Calinski-Harabasz" in cols_all:
        data["Calinski-Harabasz"] = MinMaxScaler().fit_transform(
            data["Calinski-Harabasz"].values.reshape(-1, 1)
        )
    if "Davies-Bouldin" in cols_all:
        data["Davies-Bouldin"] = MinMaxScaler().fit_transform(
            data["Davies-Bouldin"].values.reshape(-1, 1)
        )
    # Get only metric columns
    cols_metrics = [element for element in cols_all if element not in cols_id]
    # Melt columns and change column names
    data_prep = data.melt(id_vars=cols_id, value_vars=cols_metrics)
    data_prep.columns = ["model", "n_clusters", "metric", "scaled_score"]
    return data_prep


@st.cache_data(ttl=3600, max_entries=10)
def prepare_results_for_line_plot_models(data: pd.DataFrame):
    return data.groupby(by=["model", "n_clusters"]).mean().reset_index()
