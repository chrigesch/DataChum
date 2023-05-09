# Import moduls from local directories
from modules.exploratory_data_analysis.associations import cramers_v

# Import the required libraries
import math
import numpy as np
import pandas as pd
from sklearn.metrics import (
    calinski_harabasz_score,
    completeness_score,
    davies_bouldin_score,
    fowlkes_mallows_score,
    homogeneity_score,
    rand_score,
    silhouette_score,
    v_measure_score,
)

AVAILABLE_METRICS_TO_MONITOR_CLUSTERING_CONVERGENCE = (
    "Calinski-Harabasz",
    "Davies-Bouldin",
    "Silhouette",
)

AVAILABLE_METRICS_TO_MONITOR_CLUSTERING_CROSSVALIDATION_CONVERGENCE = (
    "Calinski-Harabasz",
    "Davies-Bouldin",
    "Silhouette",
    "CramersV",
    "Fowlkes-Mallows",
    "RandScore",
    "Completeness",
    "Homogeneity",
    "V-Measure",
)


def _compute_scores(
    data: pd.DataFrame,
    model_name: str,
    cluster_labels: iter,
    n_cluster: int,
):
    results_dict = {}
    results_dict["model"] = model_name
    results_dict["n_clusters"] = n_cluster
    results_dict["Calinski-Harabasz"] = calinski_harabasz_score(data, cluster_labels)
    results_dict["Davies-Bouldin"] = davies_bouldin_score(data, cluster_labels)
    results_dict["Silhouette"] = silhouette_score(data, cluster_labels)
    return results_dict


def _compute_scores_cv(
    results_dict: dict,
    cluster_labels_true: iter,
    cluster_labels_pred: iter,
):
    results_dict["CramersV"] = cramers_v(cluster_labels_true, cluster_labels_pred)[0]
    results_dict["Fowlkes-Mallows"] = fowlkes_mallows_score(
        cluster_labels_true, cluster_labels_pred
    )
    results_dict["RandScore"] = rand_score(cluster_labels_true, cluster_labels_pred)
    results_dict["Completeness"] = completeness_score(
        cluster_labels_true, cluster_labels_pred
    )
    results_dict["Homogeneity"] = homogeneity_score(
        cluster_labels_true, cluster_labels_pred
    )
    results_dict["V-Measure"] = v_measure_score(
        cluster_labels_true, cluster_labels_pred
    )
    return results_dict


def _monitor_convergence(metric_list: list, n_consecutive: int, maximize=True):
    """
    Check if the last evaluation metric in a list does not improve for n consecutive attempts.

    Args:
        metric_list (list): List of evaluation metrics.
        n_consecutive (int): Number of consecutive attempts that the evaluation metric does not improve.
        maximize (bool): If True, assume the metric is being maximized.
            If False, assume the metric is being minimized.

    Returns:
        bool: True if the last evaluation metric has not improved for n consecutive attempts, False otherwise.
    """
    if len(metric_list) <= n_consecutive:
        return False

    for i in range(1, n_consecutive + 1):
        if maximize:
            if metric_list[-i] > metric_list[-i - 1]:
                return False
        else:
            if metric_list[-i] < metric_list[-i - 1]:
                return False

    return True


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
