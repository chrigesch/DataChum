# Import moduls from local directories
from assets.colors import get_color

# Import the required libraries
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler

# Import libraries for debugging
from modules.cluster.main import clustering, clustering_cross_validation
from modules.utils.load_and_save_data import read_csv


def box_plot(
    data: pd.DataFrame,
    model_to_be_plotted: str,
    score_to_be_plotted: str,
    color: str,
):
    list_colors = get_color(
        color,
        len(data[data["model"] == model_to_be_plotted]["n_clusters"].dropna().unique()),
    )

    fig_variable = px.box(
        data,
        x="n_clusters",
        y=score_to_be_plotted,
        color_discrete_sequence=list_colors,
        color="n_clusters",
    )
    fig_variable.update_layout(xaxis_type="category", showlegend=False)
    return fig_variable


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


def prepare_results_for_line_plot_models(data: pd.DataFrame):
    return data.groupby(by=["model", "n_clusters"]).mean().reset_index()


def main():
    data = read_csv("data/data_c_and_r_with_missings.csv").drop("Loan_ID", axis=1)

    clustering_cv_instance = clustering_cross_validation(
        data=data,
        imputation_numeric="most_frequent",
        imputation_categorical="most_frequent",
        scaler="zscore",
        cluster_models=[
            "AgglomerativeClustering_single",
        ],  # 'DBSCAN' | 'AgglomerativeClustering_single' | 'SpectralClustering_nearest_neighbors' | 'SpectralClustering_rbf' | AVAILABLE_MODELS_CLUSTER
        n_cluster_min=2,
        n_cluster_max=8,
        classification_model=[
            "GaussianNaiveBayes"
        ],  # 'GaussianNaiveBayes' | 'LightGBM'
        inner_cv_folds=5,
        inner_cv_rep=1,
        n_consecutive_clusters_without_improvement=4,
        monitor_metric="Silhouette",
    )

    clustering_cv_instance.results_cv.groupby(
        by=["model", "n_clusters"]
    ).mean().sort_values(by="Silhouette", ascending=False).reset_index()

    figure_to_plot = box_plot(
        data=clustering_cv_instance.results_cv,
        model_to_be_plotted="AgglomerativeClustering_single",
        score_to_be_plotted="Silhouette",
        color="hot",  # "hot" | "magma" | "viridis"
    )

    return figure_to_plot.show()


if __name__ == "__main__":
    main()
