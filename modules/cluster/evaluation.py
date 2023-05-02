# Import moduls from local directories
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
import math
import numpy as np
import pandas as pd
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


# Import libraries for debugging


def clustering(
    data: iter,
    models_to_be_evaluated,
    n_cluster_min: int,
    n_cluster_max: int,
):
    cols_num = data.select_dtypes(include=["float", "int"]).columns.to_list()
    cols_cat = data.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.to_list()

    pipeline = data_preprocessing(
        cols_num=cols_num,
        cols_cat=cols_cat,
        imputation_numeric="mean",
        scaler="zscore",
        imputation_categorical="most_frequent",
        one_hot_encoding=True,
    )

    data_prep = pipeline.fit_transform(data)
    # Get labels of all features
    labels = _get_feature_names_after_preprocessing(pipeline, includes_model=False)
    # Convert output to Dataframe and add columns names
    data_prep = pd.DataFrame(data_prep, columns=labels, index=data.index)
    list_sil = []
    list_ch = []
    list_db = []
    list_bic = []
    list_model = []
    list_n_cluster = []
    models = cluster_models_to_evaluate(models=models_to_be_evaluated)
    for name, model in models:
        for n_cluster in range(n_cluster_min, n_cluster_max + 1):
            if name in MODELS_WITH_N_COMPONENTS:
                model.set_params(**{"n_components": n_cluster})
            elif name in MODELS_WITH_N_CLUSTER:
                model.set_params(**{"n_clusters": n_cluster})
            cluster_labels = model.fit_predict(data_prep)  # (n_cluster)
            list_sil.append(silhouette_score(data_prep, cluster_labels))
            list_ch.append(calinski_harabasz_score(data_prep, cluster_labels))
            list_db.append(davies_bouldin_score(data_prep, cluster_labels))
            list_bic.append(bic_score(data_prep.to_numpy(), cluster_labels))
            list_model.append(name)
            list_n_cluster.append(n_cluster)
    results_df = pd.DataFrame()
    results_df["model"] = list_model
    results_df["n_clusters"] = list_n_cluster
    results_df["BIC"] = list_bic
    results_df["Calinski-Harabasz"] = list_ch
    results_df["Davies_Bouldin"] = list_db
    results_df["Silhouette"] = list_sil
    return results_df


# https://github.com/smazzanti/are_you_still_using_elbow_method/blob/main/are-you-still-using-elbow-method.ipynb
def bic_score(X: np.ndarray, labels: np.array):
    """
    BIC score for the goodness of fit of clusters.
    This Python function is translated from the Golang implementation by the author of the paper.
    The original code is available here: https://github.com/bobhancock/goxmeans/blob/a78e909e374c6f97ddd04a239658c7c5b7365e5c/km.go#L778 # noqa E501
    """
    n_points = len(labels)
    n_clusters = len(set(labels))
    n_dimensions = X.shape[1]

    n_parameters = (n_clusters - 1) + (n_dimensions * n_clusters) + 1

    loglikelihood = 0
    for label_name in set(labels):
        X_cluster = X[labels == label_name]
        n_points_cluster = len(X_cluster)
        centroid = np.mean(X_cluster, axis=0)
        variance = np.sum((X_cluster - centroid) ** 2) / (len(X_cluster) - 1)
        loglikelihood += (
            n_points_cluster * np.log(n_points_cluster)
            - n_points_cluster * np.log(n_points)
            - n_points_cluster * n_dimensions / 2 * np.log(2 * math.pi * variance)
            - (n_points_cluster - 1) / 2
        )

    bic = loglikelihood - (n_parameters / 2) * np.log(n_points)

    return bic
