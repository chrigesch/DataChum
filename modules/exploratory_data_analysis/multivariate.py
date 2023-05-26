# Import moduls from local directories
from assets.colors import get_color
from modules.classification_and_regression.cv_workflow import (
    _compute_dictionary_with_scores_to_compute_in_cv,
)
from modules.utils.plots import _convert_fig_to_html
from modules.utils.preprocessing import data_preprocessing

# Import the required libraries
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LassoLarsCV, LogisticRegressionCV
import streamlit as st
from yellowbrick.features import Manifold
from yellowbrick.model_selection import DroppingCurve

# Import modules for debugging
from modules.utils.load_and_save_data import read_csv


@st.cache_data(ttl=3600, max_entries=10)
def plot_num_with_grouping_variable(
    data,
    var_num,
    var_group: str,
    barmode: str,
    color: str,
    template: str,
):
    data_to_be_plotted = (
        data[[var_group] + var_num].groupby(by=var_group).mean().reset_index()
    )
    data_to_be_plotted = pd.melt(
        data_to_be_plotted, id_vars=var_group, value_vars=var_num
    )
    if barmode in ["group", "stack"]:
        fig_variable = px.bar(
            data_to_be_plotted.sort_values(by=var_group),
            x=var_group,
            y="value",
            color="variable",
            barmode=barmode,
            color_discrete_sequence=get_color(
                color, len(data_to_be_plotted["variable"].value_counts())
            ),
            template=template,
        )
    elif barmode == "100_stack":
        fig_variable = px.histogram(
            data_to_be_plotted,
            x=var_group,
            y="value",
            color="variable",
            barnorm="percent",
            text_auto=".2f",
            color_discrete_sequence=get_color(
                color, len(data_to_be_plotted["variable"].value_counts())
            ),
            template=template,
        )
    fig_variable.update_layout(xaxis_type="category")
    fig_variable.update_layout(width=600, height=400)
    return fig_variable


@st.cache_data(ttl=3600, max_entries=10)
def plot_bubble_chart(
    data,
    x: str,
    y: str,
    var_size: str,
    var_color: str,
    var_hover_name: str,
    color: str,
    template: str,
):
    if var_color is None:
        list_colors = get_color(color, 1)
    else:
        list_colors = get_color(color, len(data[var_color].dropna().unique()))
    fig_variable = px.scatter(
        data.dropna(),
        x=x,
        y=y,
        size=var_size,
        color=var_color,
        hover_name=var_hover_name,
        color_discrete_sequence=list_colors,
        template=template,
    )
    fig_variable.update_layout(width=600, height=400)
    return fig_variable


AVAILABLE_MANIFOLD_IMPLEMENTATIONS = (
    "lle",
    "modified",
    "isomap",
    "mds",
    "spectral",
    "tsne",
)


@st.cache_data(ttl=3600, max_entries=10)
def plot_manifold(
    data,
    target_variable,
    operation,
    manifold,
    n_neighbors,
):
    # Drop all rows with NaNs in target_Variable
    data = data.dropna(subset=target_variable)
    # Separate X and y
    y = pd.Series(data[target_variable])
    X = data.drop(target_variable, axis=1)
    # Get NUMERICAL and CATEGORICAL columns
    cols_num = X.select_dtypes(include=["float", "int"]).columns.to_list()
    cols_cat = X.select_dtypes(include=["object", "category", "bool"]).columns.to_list()
    # Create preprocessing pipeline and fit_transform it
    pipeline = data_preprocessing(
        cols_num,
        cols_cat,
        imputation_numeric="mean",
        scaler="zscore",
        imputation_categorical="most_frequent",
        one_hot_encoding=True,
    )
    X = pipeline.fit_transform(X)
    # If classification problem, apply LabelEncoder to 'y' | Instantiate visualizer
    if operation == "classification":
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        visualizer = Manifold(
            scale=True,
            manifold=manifold,
            n_neighbors=n_neighbors,
            classes=label_encoder.classes_,
        )
    else:
        visualizer = Manifold(
            scale=True, manifold=manifold, n_neighbors=n_neighbors, proj_features=False
        )
    # Fit the visualizer
    visualizer.fit_transform(X, y)
    visualizer.finalize()
    fig_html = _convert_fig_to_html(visualizer.fig)
    plt.clf()
    return fig_html


@st.cache_data(ttl=3600, max_entries=10)
def plot_pca_2d(
    data,
    data_pca,
    exp_var_pc_1,
    exp_var_pc_2,
    target: str,
    color: str,
    template: str,
):
    cols_cat = data.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.to_list()
    if target in cols_cat:
        list_colors = get_color(color, len(data[target].value_counts()))
    else:
        list_colors = color
    fig_variable = px.scatter(
        data_frame=data_pca,
        x="PC1",
        y="PC2",
        color=data[target],
        color_continuous_scale=list_colors,
        color_discrete_sequence=list_colors,
        template=template,
        title="Total Explained Variance ~ "
        + "{0:.2f}".format(exp_var_pc_1 + exp_var_pc_2)
        + "%",
        labels={
            "PC1": "PC1 ~ " + "{0:.2f}".format(exp_var_pc_1) + "%",
            "PC2": "PC2 ~ " + "{0:.2f}".format(exp_var_pc_2) + "%",
        },
    )
    fig_variable.update_layout(width=600, height=400)
    return fig_variable


@st.cache_data(ttl=3600, max_entries=10)
def plot_pca_3d(
    data,
    data_pca,
    exp_var_pc_1,
    exp_var_pc_2,
    exp_var_pc_3,
    target: str,
    color: str,
    template: str,
):
    fig_variable = px.scatter_3d(
        data_frame=data_pca,
        x="PC1",
        y="PC2",
        z="PC3",
        color=data[target],
        color_discrete_sequence=get_color(color, len(data[target].value_counts())),
        template=template,
        title="Total Explained Variance ~ "
        + "{0:.2f}".format(exp_var_pc_1 + exp_var_pc_2 + exp_var_pc_3)
        + "%",
        labels={
            "PC1": "PC1 ~ " + "{0:.2f}".format(exp_var_pc_1) + "%",
            "PC2": "PC2 ~ " + "{0:.2f}".format(exp_var_pc_2) + "%",
            "PC3": "PC3 ~ " + "{0:.2f}".format(exp_var_pc_3) + "%",
        },
    )
    fig_variable.update_layout(width=600, height=400)
    return fig_variable


@st.cache_data(ttl=3600, max_entries=10)
def plot_pca_explained_variances(
    data_to_plot,
    color: str,
    template: str,
):
    fig_variable = px.line(
        data_to_plot,
        x="index",
        y="value",
        color="variable",
        markers=True,
        color_discrete_sequence=get_color(color, 2),
        template=template,
        labels={
            "index": "Principal component index",
            "value": "Explained variance ratio",
        },
    )
    fig_variable.update_layout(width=600, height=400)
    return fig_variable


@st.cache_data(ttl=3600, max_entries=10)
def plot_random_feature_dropping_curve(
    data,
    target_variable,
    operation,
    evaluation_score,
    average="macro",
    cv_folds=10,
):
    # Drop all rows with NaNs in target_Variable
    data = data.dropna(subset=target_variable)
    # Separate X and y
    y = pd.Series(data[target_variable])
    X = data.drop(target_variable, axis=1)
    # Get NUMERICAL and CATEGORICAL columns
    cols_num = X.select_dtypes(include=["float", "int"]).columns.to_list()
    cols_cat = X.select_dtypes(include=["object", "category", "bool"]).columns.to_list()
    # Create preprocessing pipeline and fit_transform it
    pipeline = data_preprocessing(
        cols_num,
        cols_cat,
        imputation_numeric="mean",
        scaler="zscore",
        imputation_categorical="most_frequent",
        one_hot_encoding=True,
    )
    X = pipeline.fit_transform(X)
    # Initiate dictionary with all available scores to compute
    scores_to_compute = _compute_dictionary_with_scores_to_compute_in_cv(
        operation, average=average
    )
    # If classification problem, apply LabelEncoder to 'y' | Instantiate visualizer
    if operation == "classification":
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        visualizer = DroppingCurve(
            LogisticRegressionCV(
                penalty="l1",
                solver="liblinear",
                cv=cv_folds,
                dual=False,
                n_jobs=-1,
                random_state=123,
            ),
            scoring=scores_to_compute[evaluation_score],
            n_jobs=-1,
            random_state=123,
        )
    else:
        visualizer = DroppingCurve(
            LassoLarsCV(cv=cv_folds, n_jobs=-1),
            scoring=scores_to_compute[evaluation_score],
            n_jobs=-1,
            random_state=123,
        )
    # Fit the visualizer
    visualizer.fit(X, y)
    visualizer.finalize()
    visualizer.ax.set(ylabel=evaluation_score)
    fig_html = _convert_fig_to_html(visualizer.fig)
    plt.clf()
    return fig_html


@st.cache_resource(ttl=3600, max_entries=10)
class principal_component_analysis:
    def __init__(self, data, target_variable):
        # Separate X
        X = data.drop(target_variable, axis=1)
        # Get NUMERICAL and CATEGORICAL columns
        cols_num = X.select_dtypes(include=["float", "int"]).columns.to_list()
        cols_cat = X.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.to_list()
        # Create preprocessing pipeline and fit_transform it
        pipeline = data_preprocessing(
            cols_num,
            cols_cat,
            imputation_numeric="mean",
            scaler="zscore",
            imputation_categorical="most_frequent",
            one_hot_encoding=True,
        )
        X = pipeline.fit_transform(X)
        col_names_without_prefix = []
        for element in pipeline.get_feature_names_out():
            element = element.removeprefix("prep_cat__")
            element = element.removeprefix("prep_num__")
            col_names_without_prefix.append(element)
        # Instantiate an PCA instance, fit it and transform data
        pca = PCA()
        data_pca = pca.fit_transform(X)
        # Create a list with the number and names of the PC
        cols_pca = [f"PC{i}" for i in range(1, pca.n_components_ + 1)]
        # Compute a DataFrame with the PCA (inspite of the features)
        self.data_pca = pd.DataFrame(data_pca, columns=cols_pca)
        # Compute a DataFrame with the weights
        self.data_weights = pd.DataFrame(
            pca.components_.T,
            columns=self.data_pca.columns,
            index=col_names_without_prefix,
        )
        # Compute a DataFrame including the explained variances
        self.data_explained_variances = pd.DataFrame(index=cols_pca).reset_index()
        self.data_explained_variances["explained"] = pca.explained_variance_ratio_
        self.data_explained_variances[
            "cumulative"
        ] = pca.explained_variance_ratio_.cumsum()
        super().__init__()


#################
# For debugging #
#################


def main():
    #    data = read_csv('/mnt/d/Proyectos/DataChum/data/data_c_and_r_with_missings.csv').drop('Loan_ID', axis=1)
    data = read_csv("/mnt/d/Proyectos/DataChum/data/data_c_and_r_complete.csv")

    plot = "pca"

    if plot == "pca":
        output = principal_component_analysis(
            data=data,
            target_variable="APOREC",  # 'Gender' | 'ApplicantIncome'
        )  # 'classification' | 'regression'
    elif plot == "random_feature_dropping":
        output = plot_random_feature_dropping_curve(
            data=data,
            target_variable="ApplicantIncome",
            operation="regression",
            evaluation_score="RMSE",
            average="macro",
            cv_folds=10,
        )
    # Convert and plot in plotly
    #    plotly_fig = tls.mpl_to_plotly(figure.fig)
    #    return iplot(plotly_fig)
    return print(output.data_pca)


if __name__ == "__main__":
    main()
