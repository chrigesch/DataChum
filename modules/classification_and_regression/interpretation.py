# Import moduls from local directories
from assets.colors import get_color
from modules.utils.preprocessing import _get_feature_names_after_preprocessing

from econml.dml import LinearDML, CausalForestDML
import fasttreeshap
import numpy as np
from optuna.visualization._utils import COLOR_SCALE
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
from sklearn.inspection import partial_dependence

# Import libraries for debugging
import joblib


def partial_dependence_plot(feature, pipeline, X, frac_ice: float):
    col_names_without_prefix = _get_feature_names_after_preprocessing(
        pipeline, includes_model=True
    )
    # Apply preprocessing to data (everything but the model)
    X_prep = pd.DataFrame(
        pipeline[:-1].transform(X),
        columns=col_names_without_prefix,
        index=X.index,
    )
    # Define maximum number of columns, instantiate subplots and set titles
    if len(feature) < 2:
        max_columns = 1
    else:
        max_columns = 2
    fig = make_subplots(
        rows=int(np.ceil(len(feature) / max_columns)),
        cols=max_columns,
        subplot_titles=(feature),
        shared_yaxes=True,
    )
    # Loop through list and add subplots
    n_col = 1
    n_row = 1
    for feat in feature:
        raw_values = partial_dependence(
            pipeline[-1], X=X_prep, features=feat, kind="both"
        )  # kind='average' | 'both'
        if frac_ice > 0:
            raw_values_ind = (
                pd.DataFrame(
                    raw_values["individual"][0], columns=raw_values["values"][0]
                )
                .sample(frac=frac_ice)
                .reset_index()
                .melt("index")
            )
            fig.add_traces(
                list(
                    px.line(
                        x=raw_values_ind["variable"],
                        y=raw_values_ind["value"],
                        color=raw_values_ind["index"],
                    ).select_traces()
                ),
                cols=n_col,
                rows=n_row,
            )
            fig.update_traces(opacity=0.15, line_color="#DCDCDC", col=n_col, row=n_row)

        fig.add_trace(
            go.Scatter(
                x=raw_values["values"][0], y=raw_values["average"][0], mode="lines"
            ),
            col=n_col,
            row=n_row,
        )
        fig.update_layout(width=600, height=400)
        n_col += 1
        if n_col > max_columns:
            n_col = 1
            n_row += 1
    fig.update_layout(showlegend=False)
    return fig


###############
# SHAP values #
###############
def compute_shap_values_agnostic(pipeline, X, n_samples: int, operation: bool):
    """Output is an instance"""
    # Get column names of the pipeline and remove prefixes
    # (do not include the model, which is the last part of the pipeline)
    col_names_without_prefix = _get_feature_names_after_preprocessing(
        pipeline, includes_model=True
    )
    # Apply preprocessing to data (everything but the model)
    X_prep = pd.DataFrame(
        pipeline[:-1].transform(X),
        columns=col_names_without_prefix,
        index=X.index,
    )
    # Set the explainer (model is the last element of the pipeline)
    if operation == "regression":
        explainer = shap.explainers.Permutation(pipeline[-1].predict, X_prep)
    elif operation == "classification":
        explainer = shap.explainers.Permutation(pipeline[-1].predict_proba, X_prep)
    # Get Shap values from preprocessed data
    shap_values = explainer(X_prep[:n_samples])  # check_additivity=False
    return shap_values


def compute_shap_values_tree(pipeline, X: iter):
    """Output is an instance"""
    # Get column names of the pipeline and remove prefixes
    # (do not include the model, which is the last part of the pipeline)
    col_names_without_prefix = _get_feature_names_after_preprocessing(
        pipeline, includes_model=True
    )
    # Apply preprocessing to data (everything but the model)
    X_prep = pd.DataFrame(pipeline[:-1].transform(X), columns=col_names_without_prefix)
    # Set the model: the last element of the pipeline
    explainer = fasttreeshap.TreeExplainer(
        pipeline[-1], n_jobs=-1
    )  # feature_perturbation='interventional'
    # Get Shap values from preprocessed data
    shap_values = explainer(X_prep, check_additivity=False)  # check_additivity=False
    return shap_values


# SHAP - GLOBAL explanations
def plot_shap_beeswarm(shap_values: iter, X: iter, color: str):
    df_plot = _create_df_to_plot_beeswarm(shap_values, X)

    fig = px.strip(
        df_plot,
        x="SHAP",
        y="Feature",
        color="Feature value",
        color_discrete_sequence=get_color(
            color, len(df_plot["Feature value"].unique())
        ),
        stripmode="overlay",
    )  # , stripmode='overlay'
    fig.update_layout(
        xaxis=dict(showgrid=True, gridcolor="WhiteSmoke", zerolinecolor="Gainsboro"),
        yaxis=dict(showgrid=True, gridcolor="WhiteSmoke", zerolinecolor="Gainsboro"),
    )
    fig.update_layout(plot_bgcolor="white", yaxis_title=None)

    # Make it so there is no gap between the supporting boxes and
    # increase the jitter so it reaches the sides of the boxes
    fig = fig.update_layout(boxgap=0).update_traces(jitter=1)
    fig.update_xaxes(showline=True, zeroline=True)
    fig.update_yaxes(automargin=True)
    # fig.update_xaxes(autorange=True)
    return fig


# Dealing with correlated features: clustering bar plot to measure observed confounders.
def plot_shap_feature_clustering(shap_values, color: str, include_dendo: bool = True):
    # Define maximum number of columns, instantiate subplots and set titles
    if include_dendo is True:
        fig = make_subplots(rows=1, cols=2, shared_yaxes=True)
        dendro_fig = ff.create_dendrogram(
            shap_values.T, orientation="left", labels=shap_values.columns
        )
        for trace in dendro_fig.select_traces():
            fig.add_trace(trace, row=1, col=2)
        dendro_leaves = dendro_fig["layout"]["yaxis"]["ticktext"][::-1]
        # Compute mean absolute feature and importance and sort values according to clustering results
        mean_abs_feat_imp_df = (
            shap_values[dendro_leaves].abs().mean(axis=0).reset_index()
        )
        mean_abs_feat_imp_df.columns = ["feature", "SHAP"]
    else:
        fig = make_subplots(rows=1, cols=1)
        # Compute mean absolute feature and importance and sort values in descending order
        mean_abs_feat_imp_df = shap_values.abs().mean(axis=0).reset_index()
        mean_abs_feat_imp_df.columns = ["feature", "SHAP"]
        mean_abs_feat_imp_df = mean_abs_feat_imp_df.sort_values(
            by="SHAP", ascending=True
        )
    # Add Bar Plot
    fig.add_traces(
        list(
            px.bar(
                data_frame=mean_abs_feat_imp_df,
                x="SHAP",
                y="feature",
                color_continuous_scale=get_color(
                    color, len(mean_abs_feat_imp_df["SHAP"].unique())
                ),
                color="SHAP",
                orientation="h",
            ).select_traces()
        ),
        cols=1,
        rows=1,
    )
    fig.update(layout_coloraxis_showscale=False)
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title="SHAP", row=1, col=1)
    fig.update_xaxes(title="Distance", row=1, col=2)
    fig.update_yaxes(automargin=True)
    return fig


# Mean Absolute Feature Importance:
def plot_shap_feature_importance(shap_values, color: str):
    shap_df = shap_values.abs().mean(axis=0).sort_values().reset_index()
    shap_df.columns = ["feature", "SHAP"]
    fig = px.bar(
        data_frame=shap_df.sort_values(by="SHAP", ascending=True),
        x="SHAP",
        y="feature",
        color="SHAP",
        color_continuous_scale=get_color(color, len(shap_df["SHAP"].unique())),
        orientation="h",
    )
    fig.update_layout(width=600, height=400, yaxis_title=None)
    fig.update_yaxes(automargin=True)
    return fig


# Dependence scatter plot: to show the effect of a single feature across the whole dataset
# Dependence scatter plot: to show the effect of a single feature across the whole dataset
def plot_shap_scatter(
    shap_values: iter,
    X: iter,
    feature: str,
    color: str,
    var_color: str = None,
):
    """Parameters
    shap_values: iter
        Must be a DataFrame or"""
    # Create a list of colors for the markers
    if var_color is None:
        list_colors = get_color(color, 1)[0]
    else:
        list_colors = X[var_color]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=X[feature],
            y=shap_values[feature],
            mode="markers",
            yaxis="y",
            marker=dict(
                size=6,
                color=list_colors,  # set color equal to a variable: list_colors | OR use a colorname | '#DCDCDC'
                colorscale=color,
            ),
            hovertemplate=str(feature) + ": %{x:.3f}" + "<br>SHAP: %{y:.3f}",
        )
    )  # one of plotly colorscales: 'Viridis'
    if var_color is not None:
        fig.update_traces(
            marker=dict(
                showscale=True,
                colorbar=dict(
                    title=str(var_color), orientation="v", titleside="top", x=1.15
                ),
            ),
            hovertemplate=str(feature)
            + ": %{x:.3f}"
            + "<br>SHAP: %{y:.3f}<br>"
            + str(var_color)
            + " %{marker.color:.3f}",
        )

    fig.add_trace(
        go.Histogram(
            x=X[feature],
            yaxis="y2",
            histnorm="percent",
            opacity=0.1,
            hovertemplate="Interval: " + "%{x}" + "<br>Percent:" + "%{y:.2f}%",
        )
    )

    fig.update_layout(
        showlegend=False,
        xaxis=dict(title=feature),
        yaxis=dict(
            title=dict(text="SHAP values"),
            side="left",
            range=[min(shap_values[feature]), max(shap_values[feature])],
        ),
        yaxis2=dict(
            title=dict(text="Histogram values (%)"),
            side="right",
            range=[0, 100],
            overlaying="y",
            tickmode="sync",
        ),
    )
    fig.update_xaxes(showline=True, zeroline=False)
    fig.update_yaxes(showline=True, zeroline=False)
    fig.update_layout(width=600, height=400)

    return fig


# LOCAL explanations (Feature importance plot: mean absolute value of each feature)
def plot_shap_local(shap_values, id: int, color: str):
    shap_df = shap_values.iloc[id].reset_index()
    shap_df.columns = ["feature", "SHAP"]
    fig = px.bar(
        data_frame=shap_df.sort_values(by="SHAP", ascending=True),
        x="SHAP",
        y="feature",
        color="SHAP",
        color_continuous_scale=get_color(color, len(shap_df["SHAP"].unique())),
        orientation="h",
    )
    fig.update_yaxes(automargin=True)
    fig.update_layout(yaxis_title=None)
    return fig


# Double/Debiased ML: Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018). Double/debiased machine learning for treatment and structural parameters. The Econometrics Journal, 21(1), C1–C68. https://doi.org/10.1111/ectj.12097  # noqa: E501
# Causal Forest: Wager, S., & Athey, S. (2018). Estimation and Inference of Heterogeneous Treatment Effects using Random Forests. Journal of the American Statistical Association, 113(523), 1228–1242. https://doi.org/10.1080/01621459.2017.1319839  # noqa: E501
AVAILABLE_METHODS_FOR_DOUBLE_ML = ["custom", "linear", "forest"]


def compute_average_treatment_effect(
    pipeline, data, target_variable, estimation_method, operation, label_encoder=None
):
    assert estimation_method in AVAILABLE_METHODS_FOR_DOUBLE_ML, (
        "Unrecognized value, 'estimation_method' should be one of the following: "
        + str(AVAILABLE_METHODS_FOR_DOUBLE_ML)
    )
    # Drop all rows with NaNs in target_Variable
    data = data.dropna(subset=target_variable)
    # Separate X and y
    y = pd.Series(data[target_variable])
    X = data.drop(target_variable, axis=1)
    # If classification problem, apply LabelEncoder to 'y'
    if operation == "classification":
        y = label_encoder.transform(y)
    else:
        pass
    # Get column names of the pipeline and remove prefixes
    # (do not include the model, which is the last part of the pipeline)
    col_names_without_prefix = _get_feature_names_after_preprocessing(
        pipeline, includes_model=True
    )
    # Apply preprocessing to data (everything but the model)
    observations = pd.DataFrame(
        pipeline[:-1].transform(X), columns=col_names_without_prefix
    )
    # Initiate variables to collect results
    feature_name = []
    ate = []
    p_value = []
    # Loop through X to compute p values of each feature
    for causal_feature_str in observations.columns.to_list():
        causal_feature = pd.Series(observations[causal_feature_str])
        control_features = observations.drop(causal_feature_str, axis=1)
        if estimation_method == "custom":
            est = LinearDML(cv=5, model_y=pipeline[-1], random_state=123)
            est.fit(y, T=causal_feature, W=control_features)
            feature_name.append(causal_feature_str)
            ate.append(est.ate())  # X=control_features | heterogeneous treatment effect
            p_value.append(est.ate_inference().pvalue())
        elif estimation_method == "linear":
            est = LinearDML(cv=5, random_state=123)
            est.fit(y, T=causal_feature, W=control_features)
            feature_name.append(causal_feature_str)
            ate.append(est.ate())
            p_value.append(est.ate_inference().pvalue())
        else:
            est = CausalForestDML(cv=5, random_state=123)
            est.fit(y, T=causal_feature, X=control_features)
            feature_name.append(causal_feature_str)
            ate.append(est.ate(X=control_features))
            p_value.append(est.ate_inference(X=control_features).pvalue())

    results_causal_features = pd.DataFrame()
    results_causal_features["feature"] = feature_name
    results_causal_features["ate"] = ate
    results_causal_features["p_value"] = p_value
    return results_causal_features


def plot_ate(ate_df):
    fig = px.bar(
        data_frame=ate_df.sort_values(by="ate", ascending=False),
        x="feature",
        y="ate",
        color="ate",
        color_discrete_sequence=COLOR_SCALE,
    )
    fig.update_layout(width=600, height=400, yaxis_title="Average Treament Effects")
    return fig


######################################
# Private Methods / Helper functions #
######################################
def _create_df_to_plot_beeswarm(shap_values, X):
    # Initiate Dataframe
    df_to_plot = pd.DataFrame()
    X_temp = X.copy(deep=True)
    # Loop through both Dataframes (ordered by absolute mean feature importance), join and melt
    for column in shap_values.abs().mean(axis=0).sort_values().index:
        df_temp = pd.DataFrame()
        mean_column = np.mean(X[column])
        X_temp.loc[X_temp[column] <= mean_column, "new"] = "low"
        X_temp.loc[X_temp[column] > mean_column, "new"] = "high"
        df_temp["Feature value"] = X_temp["new"]
        df_temp["SHAP"] = shap_values[column]
        df_temp["Feature"] = column
        df_to_plot = pd.concat([df_to_plot, df_temp], axis=0)
    return df_to_plot


#################
# For debugging #
#################
def main():
    # Import data, pipeline and label encoder
    data = pd.read_csv("data/data_c_and_r_with_missings.csv").drop("Loan_ID", axis=1)
    pipeline = joblib.load("data/pipeline_LBGMClassifier_Gender")
    # label_encoder = joblib.load("../../data/label_encoder_Gender")

    # results_double_ml = compute_average_treatment_effect(
    # pipeline=pipeline,
    # data=data,
    # target_variable="Gender",
    # estimation_method="forest",
    # operation="classification",
    # label_encoder=label_encoder,
    # )

    target_variable = "Gender"
    X_to_be_used = data.drop(target_variable, axis=1)

    shap_values_instance = compute_shap_values_tree(pipeline=pipeline, X=X_to_be_used)

    sv_to_be_plotted = pd.DataFrame(
        shap_values_instance.values,
        columns=shap_values_instance.feature_names,
    )
    plot = plot_shap_scatter(
        shap_values=sv_to_be_plotted,
        X=X_to_be_used,
        feature="CoapplicantIncome",  # ['CoapplicantIncome','LoanAmount']
        color="magma",
        var_color=None,
    )
    return plot.show()


if __name__ == "__main__":
    main()
