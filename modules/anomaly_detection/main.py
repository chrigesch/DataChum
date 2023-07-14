# Import moduls from local directories
from modules.anomaly_detection.models import anomaly_detection_models_to_evaluate
from modules.classification_and_regression.metrics import (
    _compute_dictionary_with_scores_to_compute_in_cv,
)
from modules.classification_and_regression.models import regression_models_to_tune
from modules.utils.preprocessing import data_preprocessing

# Import the required Libraries
import pandas as pd
from sklearn.model_selection import RepeatedKFold
from time import time


class anomaly_detection_cross_validation:
    def __init__(
        self,
        data: iter,
        imputation_numerical: str,
        imputation_categorical: str,
        scaler: str,
        anomaly_detection_models: list,
        regression_model: list,
        inner_cv_folds: int,
        inner_cv_rep: int,
    ):
        self.method = "k-fold"
        self.imputation_numerical = imputation_numerical
        self.imputation_categorical = imputation_categorical
        self.scaler = scaler
        self.anomaly_detection_models = anomaly_detection_models
        self.regression_model = regression_model
        self.inner_cv_folds = inner_cv_folds
        self.inner_cv_rep = inner_cv_rep
        # Remove data duplicates while retaining the first one
        data = data.drop_duplicates(keep="first", inplace=False)
        # Get categorical and numerical column names
        cols_num = data.select_dtypes(include=["float", "int"]).columns.to_list()
        cols_cat = data.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.to_list()
        # Initiate list to collect the results
        self.all_results = pd.DataFrame()
        # Get list of models
        anomaly_detection_models_list = anomaly_detection_models_to_evaluate(
            models=self.anomaly_detection_models
        )
        for (
            name_anomaly_detection_model,
            anomaly_detection_model,
        ) in anomaly_detection_models_list:
            # Compute baseline prediction strength
            # Instantiante an cross-validation instance
            inner_cv_object = RepeatedKFold(
                n_splits=self.inner_cv_folds,
                n_repeats=self.inner_cv_rep,
                random_state=123,
            )
            # Start inner loop for cross-validation
            for counter, (train_index, test_index) in enumerate(
                inner_cv_object.split(data),
                start=1,
            ):
                time_start = time()
                # Split training and testing data
                X_train, X_val = (
                    data.loc[data.index[train_index]],
                    data.loc[data.index[test_index]],
                )
                # Create pipeline for data preparation
                self.pipeline = data_preprocessing(
                    cols_num=cols_num,
                    cols_cat=cols_cat,
                    imputation_numerical=self.imputation_numerical,
                    scaler=self.scaler,
                    imputation_categorical=self.imputation_categorical,
                    one_hot_encoding=True,
                )
                # Prepare data
                X_train_prep = self.pipeline.fit_transform(X_train)
                X_val_prep = self.pipeline.transform(X_val)
                # Fit a anomaly detection model on the train data and make predictions for it
                y_train = anomaly_detection_model.fit(X_train_prep).predict_proba(
                    X_train_prep,
                    method="unify",
                )[:, 1]
                # Fit a anomay detection model on the validation data and make predictions for it
                y_val = anomaly_detection_model.fit(X_val_prep).predict_proba(
                    X_val_prep,
                    method="unify",
                )[:, 1]
                # Fit the prediction model on the "complete" train data
                prediction_model = regression_models_to_tune(
                    models=self.regression_model,
                    cv_with_pipeline=False,
                    n_rows=X_train_prep.shape[0],
                    n_cols=X_train_prep.shape[1],
                    inner_cv_folds=inner_cv_folds,
                )[0][2]
                # Fit prediction model
                prediction_model.fit(X_train_prep, y_train)
                # Initiate dictionary with all scores to compute
                scores_to_compute = _compute_dictionary_with_scores_to_compute_in_cv(
                    "regression",
                    average="macro",
                )
                time_end = time()
                time_total = time_end - time_start
                # Use the fitted prediction model to compute predictions for validation data
                # Append all scores to results
                validation_scores = _compute_scores_anomaly_detection(
                    anomaly_detection_model_name=name_anomaly_detection_model,
                    time=time_total,
                    scores_to_compute=scores_to_compute,
                    fitted_model=prediction_model,
                    X=X_val_prep,
                    y_true=y_val,
                )
                self.all_results = pd.concat(
                    [self.all_results, validation_scores], axis=0
                )
            print("Finished", name_anomaly_detection_model)


class anomaly_detection_standart:
    def __init__(
        self,
        data: iter,
        imputation_numerical: str,
        imputation_categorical: str,
        scaler: str,
        anomaly_detection_models: list,
    ):
        self.method = "standart"
        self.imputation_numerical = imputation_numerical
        self.imputation_categorical = imputation_categorical
        self.scaler = scaler
        self.anomaly_detection_models = anomaly_detection_models


######################################
# Private Methods / Helper functions #
######################################


def _compute_scores_anomaly_detection(
    anomaly_detection_model_name,
    time,
    scores_to_compute,
    fitted_model,
    X,
    y_true,
):
    # Create an empty dictionary to collect the results and add the adapted model name
    results_dict = {}
    results_dict["model"] = anomaly_detection_model_name
    results_dict["time"] = time
    # Compute all scores
    for score_name in scores_to_compute:
        score_function = scores_to_compute[score_name]
        results_dict[score_name] = abs(float(score_function(fitted_model, X, y_true)))
    # Convert dictionary to DataFrame
    results_scores_test = pd.DataFrame.from_dict([results_dict])

    return results_scores_test
