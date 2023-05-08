# Import moduls from local directories
from modules.classification_and_regression.models import classification_models_to_tune
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
    fowlkes_mallows_score,
    silhouette_score,
)
from sklearn.model_selection import RepeatedKFold
from sklearn.utils import resample

# Import libraries for debugging


def clustering(
    data: iter,
    imputation_numeric: str,
    imputation_categorical: str,
    scaler: str,
    models_to_be_evaluated: list,
    n_cluster_min: int,
    n_cluster_max: int,
    n_bootstrap_samples: int,
):
    cols_num = data.select_dtypes(include=["float", "int"]).columns.to_list()
    cols_cat = data.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.to_list()

    pipeline = data_preprocessing(
        cols_num=cols_num,
        cols_cat=cols_cat,
        imputation_numeric=imputation_numeric,
        scaler=scaler,
        imputation_categorical=imputation_categorical,
        one_hot_encoding=True,
    )

    data_prep = pipeline.fit_transform(data)
    # Get labels of all features
    labels = _get_feature_names_after_preprocessing(pipeline, includes_model=False)
    # Convert output to Dataframe and add columns names
    data_prep = pd.DataFrame(data_prep, columns=labels, index=data.index)
    # Initiate list to collect the results
    results_list = []
    # Get list of models
    models = cluster_models_to_evaluate(models=models_to_be_evaluated)
    for name, model in models:
        if (name == "DBSCAN") & (n_bootstrap_samples == 0):
            # Fit the model, compute and append the scores
            cluster_labels = model.fit_predict(data_prep)
            results_dict = _compute_scores(
                data=data_prep,
                model_name=name,
                cluster_labels=cluster_labels,
                n_cluster=len(np.unique(model.labels_)),
            )
            results_list.append(results_dict)
        elif (name == "DBSCAN") & (n_bootstrap_samples > 0):
            coefficients_of_variation_list = []
            for n_bootstrap in range(1, n_bootstrap_samples + 1):
                # Create a resampled DataFrame
                data_bootstrap = resample(
                    data_prep, replace=True, random_state=n_bootstrap
                )
                # Fit the model, compute and append the scores
                cluster_labels = model.fit_predict(data_bootstrap)
                results_dict = _compute_scores(
                    data=data_bootstrap,
                    model_name=name,
                    cluster_labels=cluster_labels,
                    n_cluster=len(np.unique(model.labels_)),
                )
                results_list.append(results_dict)
                # Every 50 bootstrap replicates, monitor convergence and stop if there is no improvement
                # in the Silhouette score
                if n_bootstrap % 50 == 0:
                    results_temp = pd.DataFrame.from_dict(results_list)
                    coefficients_of_variation_temp = abs(
                        np.std(results_temp["Silhouette"])
                        / np.mean(results_temp["Silhouette"])
                    )
                    coefficients_of_variation_list.append(
                        coefficients_of_variation_temp
                    )
                    print("Bootstrap: ", n_bootstrap)
                    print("cv: ", coefficients_of_variation_temp)
                    if _monitor_convergence(coefficients_of_variation_list, 3, False):
                        break
        else:
            for n_cluster in range(n_cluster_min, n_cluster_max + 1):
                if name in MODELS_WITH_N_COMPONENTS:
                    model.set_params(**{"n_components": n_cluster})
                elif name in MODELS_WITH_N_CLUSTER:
                    model.set_params(**{"n_clusters": n_cluster})
                # Compute the scores
                if n_bootstrap_samples == 0:
                    # Fit the model and append the scores
                    cluster_labels = model.fit_predict(data_prep)
                    results_dict = _compute_scores(
                        data=data_prep,
                        model_name=name,
                        cluster_labels=cluster_labels,
                        n_cluster=n_cluster,
                    )
                    results_list.append(results_dict)
                else:
                    coefficients_of_variation_list = []
                    for n_bootstrap in range(1, n_bootstrap_samples + 1):
                        # Create a resampled DataFrame
                        data_bootstrap = resample(
                            data_prep, replace=True, random_state=n_bootstrap
                        )
                        # Fit the model and append the scores
                        cluster_labels = model.fit_predict(data_bootstrap)
                        results_dict = _compute_scores(
                            data=data_bootstrap,
                            model_name=name,
                            cluster_labels=cluster_labels,
                            n_cluster=n_cluster,
                        )
                        results_list.append(results_dict)
                        # Every 50 bootstrap replicates, monitor convergence and stop if there is no improvement
                        # in the Silhouette score
                        # Citation 50 bootstrap replicates criterion: Pattengale, N. D., Alipour, M., Bininda-Emonds, O. R. P., Moret, B. M. E., & Stamatakis, A. (2010). How Many Bootstrap Replicates Are Necessary? Journal of Computational Biology, 17(3), 337–354. https://doi.org/10.1089/cmb.2009.0179 # noqa E501
                        if n_bootstrap % 50 == 0:
                            results_temp = pd.DataFrame.from_dict(results_list)
                            coefficients_of_variation_temp = abs(
                                np.std(results_temp["Silhouette"])
                                / np.mean(results_temp["Silhouette"])
                            )
                            coefficients_of_variation_list.append(
                                coefficients_of_variation_temp
                            )
                            if _monitor_convergence(
                                coefficients_of_variation_list, 3, False
                            ):
                                break
    # Convert the list of dictionaries to DataFrame
    results_df = pd.DataFrame.from_dict(results_list)
    return results_df


# Prediction-based resampling method:
# Dudoit, S., & Fridlyand, J. (2002). A prediction-based resampling method for estimating the number
# of clusters in a dataset. Genome Biology, 3(7), 1–21.
def clustering_cross_validation(
    data: iter,
    imputation_numeric: str,
    imputation_categorical: str,
    scaler: str,
    cluster_models: list,
    n_cluster_min: int,
    n_cluster_max: int,
    classification_model: list,
    inner_cv_folds: int,
    inner_cv_rep: int,
):
    # Assert input values
    for cluster_model in cluster_models:
        assert (
            cluster_model != "DBSCAN"
        ), "As the number of clusters cannot be preasigned, DBSCAN ist no sopported for cross-validation"
    # Remove data duplicates while retaining the first one
    data = data.drop_duplicates(keep="first", inplace=False)
    # Get categorical and numerical column names
    cols_num = data.select_dtypes(include=["float", "int"]).columns.to_list()
    cols_cat = data.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.to_list()
    # Initiate list to collect the results
    results_list = []
    # Get list of models
    cluster_models_list = cluster_models_to_evaluate(models=cluster_models)
    for name_cluster_model, cluster_model in cluster_models_list:
        for n_cluster in range(n_cluster_min, n_cluster_max + 1):
            if name_cluster_model in MODELS_WITH_N_COMPONENTS:
                cluster_model.set_params(**{"n_components": n_cluster})
            elif name_cluster_model in MODELS_WITH_N_CLUSTER:
                cluster_model.set_params(**{"n_clusters": n_cluster})

            # Compute baseline prediction strength
            # Instantiante an cross-validation instance
            inner_cv_object = RepeatedKFold(
                n_splits=inner_cv_folds,
                n_repeats=inner_cv_rep,
                random_state=123,
            )
            # Start inner loop for cross-validation
            for counter, (train_index, test_index) in enumerate(
                inner_cv_object.split(data),
                start=1,
            ):
                # Split training and testing data
                X_train, X_val = (
                    data.loc[data.index[train_index]],
                    data.loc[data.index[test_index]],
                )
                # Create pipeline for data preparation
                pipeline = data_preprocessing(
                    cols_num=cols_num,
                    cols_cat=cols_cat,
                    imputation_numeric=imputation_numeric,
                    scaler=scaler,
                    imputation_categorical=imputation_categorical,
                    one_hot_encoding=True,
                )
                # Prepare data
                X_train_prep = pipeline.fit_transform(X_train)
                X_val_prep = pipeline.transform(X_val)
                # Fit a cluster model on the train data and make predictions for it
                y_train = cluster_model.fit_predict(X_train_prep)
                # Fit a cluster model on the validation data and make predictions for it
                y_val = cluster_model.fit_predict(X_val_prep)
                # Fit the prediction model on the "complete" train data
                prediction_model = classification_models_to_tune(
                    models=classification_model,
                    cv_with_pipeline=False,
                    n_rows=X_train_prep.shape[0],
                    n_cols=X_train_prep.shape[1],
                    inner_cv_folds=inner_cv_folds,
                )[0][2]
                prediction_model.fit(X_train_prep, y_train)
                # Use the fitted prediction model to compute predictions for validation data
                y_pred = prediction_model.predict(X_val_prep)
                # Compute prediction strength
                fowlkes_mallows = fowlkes_mallows_score(y_val, y_pred)
                # Append all scores to results
                results_dict = _compute_scores(
                    data=X_train_prep,
                    model_name=name_cluster_model,
                    cluster_labels=y_train,
                    n_cluster=n_cluster,
                )
                results_dict["Fowlkes-Mallows"] = fowlkes_mallows
                results_list.append(results_dict)

            print("Finished", name_cluster_model, "- n_cluster:", n_cluster)

    # Convert the list of dictionaries to DataFrame
    all_results_cv = pd.DataFrame.from_dict(results_list)
    return all_results_cv


######################################
# Private Methods / Helper functions #
######################################


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
