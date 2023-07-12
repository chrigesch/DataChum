# Import moduls from local directories
from assets.colors import get_color
from modules.anomaly_detection.models import anomaly_detection_models_to_evaluate
from modules.utils.preprocessing import data_preprocessing

# Import the required Libraries
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots


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


def get_anomaly_scores(
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

    # Get list of model
    anomaly_detection_model_list = anomaly_detection_models_to_evaluate(
        models=[anomaly_detection_model]
    )
    anomaly_scores_min_max = (
        anomaly_detection_model_list[0][1].fit(data_prep).predict_proba(data_prep)[:, 1]
    )
    return anomaly_scores_min_max
