# Import moduls from local directories
from assets.colors import get_color

# Import the required libraries
import pandas as pd
import plotly.express as px

# Import libraries for debugging
from modules.cluster.main import clustering, clustering_cross_validation
from modules.utils.load_and_save_data import read_csv


def box_plot(
    data: pd.DataFrame, model_to_be_plotted: str, score_to_be_plotted: str, color: str
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
    fig_variable.update(layout_coloraxis_showscale=False)
    fig_variable.update_layout(xaxis_type="category")
    fig_variable.update_layout(width=600, height=400)
    return fig_variable


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
        n_consecutive_clusters_without_improvement=5,
        monitor_metric="Silhouette",
    )

    clustering_cv_instance.results_cv.groupby(
        by=["model", "n_clusters"]
    ).mean().sort_values(by="Silhouette", ascending=False).reset_index()

    figure_to_plot = box_plot(
        data=clustering_cv_instance.results_cv,
        model_to_be_plotted="AgglomerativeClustering_single",
        score_to_be_plotted="Silhouette",
        color="viridis",
    )

    return figure_to_plot.show()


if __name__ == "__main__":
    main()
