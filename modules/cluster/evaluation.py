# Import moduls from local directories
from modules.classification_and_regression.models import classification_models_to_tune
from modules.cluster.metrics import (
    _compute_scores,
    _compute_scores_cv,
    _monitor_convergence,
)
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
from sklearn.model_selection import RepeatedKFold
from sklearn.utils import resample

# Import libraries for debugging


class clustering:
    def __init__(
        self,
        data: iter,
        imputation_numeric: str,
        imputation_categorical: str,
        scaler: str,
        models_to_be_evaluated: list,
        n_cluster_min: int,
        n_cluster_max: int,
        n_bootstrap_samples: int,
        n_consecutive_bootstraps_without_improvement: int,
        n_consecutive_clusters_without_improvement: int,
        monitor_metric: str,
    ):
        self.models_to_be_evaluated = models_to_be_evaluated
        self.monitor_metric = monitor_metric
        cols_num = data.select_dtypes(include=["float", "int"]).columns.to_list()
        cols_cat = data.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.to_list()

        self.pipeline = data_preprocessing(
            cols_num=cols_num,
            cols_cat=cols_cat,
            imputation_numeric=imputation_numeric,
            scaler=scaler,
            imputation_categorical=imputation_categorical,
            one_hot_encoding=True,
        )

        data_prep = self.pipeline.fit_transform(data)
        # Get labels of all features
        labels = _get_feature_names_after_preprocessing(
            self.pipeline,
            includes_model=False,
        )
        # Convert output to Dataframe and add columns names
        data_prep = pd.DataFrame(data_prep, columns=labels, index=data.index)
        # Initiate list to collect the results
        results_list = []
        # Get list of models
        models = cluster_models_to_evaluate(models=self.models_to_be_evaluated)
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
                    if (n_consecutive_bootstraps_without_improvement is not None) & (
                        n_bootstrap % 50 == 0
                    ):
                        results_temp = pd.DataFrame.from_dict(results_list)
                        coefficients_of_variation_temp = abs(
                            np.std(results_temp[self.monitor_metric])
                            / np.mean(results_temp[self.monitor_metric])
                        )
                        coefficients_of_variation_list.append(
                            coefficients_of_variation_temp
                        )
                        print("Bootstrap: ", n_bootstrap)
                        print("cv: ", coefficients_of_variation_temp)
                        if _monitor_convergence(
                            coefficients_of_variation_list,
                            n_consecutive_bootstraps_without_improvement,
                            False,
                        ):
                            break
            else:
                monitor_metrics_per_cluster_list = []
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
                        monitor_coefficients_of_variation_list = []
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
                            # in the coefficients of variation of the selected metric
                            # Citation 50 bootstrap replicates criterion: Pattengale, N. D., Alipour, M., Bininda-Emonds, O. R. P., Moret, B. M. E., & Stamatakis, A. (2010). How Many Bootstrap Replicates Are Necessary? Journal of Computational Biology, 17(3), 337–354. https://doi.org/10.1089/cmb.2009.0179 # noqa E501
                            if (
                                n_consecutive_bootstraps_without_improvement is not None
                            ) & (n_bootstrap % 50 == 0):
                                results_temp = pd.DataFrame.from_dict(results_list)
                                coefficients_of_variation_temp = abs(
                                    np.std(results_temp[self.monitor_metric])
                                    / np.mean(results_temp[self.monitor_metric])
                                )
                                monitor_coefficients_of_variation_list.append(
                                    coefficients_of_variation_temp
                                )
                                if _monitor_convergence(
                                    monitor_coefficients_of_variation_list,
                                    n_consecutive_bootstraps_without_improvement,
                                    False,
                                ):
                                    break
                        # Monitor convergence of adding clusters and stop
                        # if there is no improvement in the selected metric
                        if n_consecutive_clusters_without_improvement is not None:
                            if self.monitor_metric == "Davies-Bouldin":
                                maximize = False
                            else:
                                maximize = True
                            # Create a DataFrame and filter only the specific model n_clusters
                            df_temp = pd.DataFrame.from_dict(results_list)
                            metric_mean = df_temp[
                                (df_temp["model"] == name)
                                & (df_temp["n_clusters"] == n_cluster)
                            ][self.monitor_metric].mean()
                            monitor_metrics_per_cluster_list.append(metric_mean)
                            if _monitor_convergence(
                                monitor_metrics_per_cluster_list,
                                n_consecutive_clusters_without_improvement,
                                maximize,
                            ):
                                break
                    print(
                        "Finished",
                        name,
                        "- n_cluster:",
                        n_cluster,
                        "- bootstrap samples: ",
                        n_bootstrap,
                    )
        # Convert the list of dictionaries to DataFrame
        self.all_results = pd.DataFrame.from_dict(results_list)


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
    n_consecutive_clusters_without_improvement: int,
    monitor_metric: str,
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
        # Initiate list to collect the results
        monitor_metrics_per_cluster_list = []
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
                # Append all scores to results
                results_dict = _compute_scores(
                    data=X_train_prep,
                    model_name=name_cluster_model,
                    n_cluster=n_cluster,
                    cluster_labels=y_train,
                )
                results_dict = _compute_scores_cv(
                    results_dict=results_dict,
                    cluster_labels_pred=y_pred,
                    cluster_labels_true=y_val,
                )
                results_list.append(results_dict)

            # Monitor convergence of adding clusters and stop if there is no improvement in the selected metric
            if n_consecutive_clusters_without_improvement is not None:
                if monitor_metric == "Davies-Bouldin":
                    maximize = False
                else:
                    maximize = True
                # Create a DataFrame and filter only the specific model and n_clusters
                df_temp = pd.DataFrame.from_dict(results_list)
                metric_mean = df_temp[
                    (df_temp["model"] == name_cluster_model)
                    & (df_temp["n_clusters"] == n_cluster)
                ][monitor_metric].mean()
                monitor_metrics_per_cluster_list.append(metric_mean)
                if (
                    _monitor_convergence(
                        monitor_metrics_per_cluster_list,
                        n_consecutive_clusters_without_improvement,
                        maximize,
                    )
                    is True
                ):
                    break

            print("Finished", name_cluster_model, "- n_cluster:", n_cluster)

    # Convert the list of dictionaries to DataFrame
    all_results_cv = pd.DataFrame.from_dict(results_list)
    return all_results_cv


######################################
# Private Methods / Helper functions #
######################################
