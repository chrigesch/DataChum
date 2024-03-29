# Import moduls from local directories
from assets.colors import get_color
from modules.anomaly_detection.models import anomaly_detection_models_to_evaluate
from modules.utils.preprocessing import (
    data_preprocessing,
    _get_feature_names_after_preprocessing,
)

# Import the required Libraries
import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st


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
def plot_anomalies_evaluation(
    outlier_scores,
    name_model: str,
    color: str,
):
    # Create a list of colors for the markers
    list_colors = get_color(color, 1)[0]
    fig = make_subplots(
        rows=2, cols=1, specs=[[{"secondary_y": True}], [{"secondary_y": True}]]
    )
    fig.add_trace(
        px.ecdf(
            outlier_scores,
            markers=True,
            lines=False,
            color_discrete_sequence=[list_colors],
        ).data[0],
        row=2,
        col=1,
        secondary_y=False,
    )

    fig.add_trace(
        px.histogram(outlier_scores, opacity=0.2, histnorm="percent").data[0],
        row=2,
        col=1,
        secondary_y=True,
    )

    fig.add_trace(
        px.box(x=outlier_scores, color_discrete_sequence=[list_colors]).data[0],
        row=1,
        col=1,
        secondary_y=False,
    )
    # Set y-axes titles
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(title="MinMax-scaled Anomaly Scores", row=2, col=1)
    fig.update_yaxes(
        title_text="ECDF - probability", row=2, col=1, secondary_y=False, range=[0, 1.1]
    )
    fig.update_yaxes(
        title_text="Histogram - percent", row=2, col=1, secondary_y=True, showgrid=False
    )
    fig.update_layout(title_text=name_model, showlegend=False)
    return fig


######################################
# Private Methods / Helper functions #
######################################


@st.cache_data(ttl=3600, max_entries=10)
def get_anomaly_scores_and_data_prep(
    data: pd.DataFrame,
    imputation_numerical: str,
    imputation_categorical: str,
    scaler: str,
    anomaly_detection_model: str,
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

    # Get list of model
    anomaly_detection_model_list = anomaly_detection_models_to_evaluate(
        models=[anomaly_detection_model]
    )
    anomaly_scores_min_max = (
        anomaly_detection_model_list[0][1]
        .fit(data_prep)
        .predict_proba(
            data_prep,
            method="linear",
        )[:, 1]
    )
    return pd.Series(anomaly_scores_min_max, name="anomaly_score"), data_prep


@st.cache_data(ttl=3600, max_entries=10)
def select_cases_for_line_plot(
    data_prep: pd.DataFrame,
    anomaly_scores: np.array,
    threshold: float,
):
    data_prep["anomaly_score"] = anomaly_scores
    selected_cases = data_prep[data_prep["anomaly_score"] >= threshold]
    selected_cases.loc["median_all"] = data_prep.median()
    selected_cases.loc["mean_all"] = data_prep.mean()

    return selected_cases.reset_index().melt(id_vars="index")
