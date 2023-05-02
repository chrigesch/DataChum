# Import moduls from local directories
from assets.strings import in_classification_and_regression as string
from modules.classification_and_regression.feature_selection import (
    add_feature_selection_to_pipeline,
)
from modules.classification_and_regression.models import (
    regression_models_to_tune,
    classification_models_to_tune,
)
from modules.utils.preprocessing import (
    data_preprocessing,
    _get_feature_names_after_preprocessing,
    tune_data_preprocessing,
)

# Import the required libraries
from copy import deepcopy
import numpy as np
import pandas as pd
from pandas import qcut
from statistics import mean
from time import time
from sklearn.model_selection import RepeatedStratifiedKFold
from optuna.integration import OptunaSearchCV
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    recall_score,
    precision_score,
    f1_score,
    cohen_kappa_score,
    matthews_corrcoef,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import make_pipeline

# Import libraries for debugging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

pd.options.display.float_format = "{:.3f}".format

AVAILABLE_OPERATIONS = ["classification", "regression"]
AVAILABLE_SCORES_REGRESSION = ["MAE", "MSE", "RMSE", "RMSLE", "R2", "MAPE"]
AVAILABLE_SCORES_CLASSIFICATION = [
    "Accuracy",
    "AUC",
    "Recall",
    "Precision",
    "F1",
    "Kappa",
    "MCC",
]
AVAILABLE_NUMBER_OF_INNER_CV_FOLDS = list(range(5, 11))


def _cv_preparation(
    X_train,
    X_test,
    operation,
    inner_cv_folds,
    imputation_numeric,
    imputation_categorical,
    scaler,
    one_hot_encoding,
    feature_selection,
    models_to_be_evaluated,
    cv_with_pipeline,
):
    # Reset index
    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    # Get NUMERICAL and CATEGORICAL columns
    cols_num = X_train.select_dtypes(include=["float", "int"]).columns.to_list()
    cols_cat = X_train.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.to_list()
    # Data preparation
    pipeline = data_preprocessing(
        cols_num,
        cols_cat,
        imputation_numeric=imputation_numeric,
        scaler=scaler,
        imputation_categorical=imputation_categorical,
        one_hot_encoding=one_hot_encoding,
    )
    # Add feature selection to pipeline
    pipeline = add_feature_selection_to_pipeline(
        pipeline=pipeline, operation=operation, method=feature_selection
    )
    # Select and add models
    if operation == "regression":
        models = regression_models_to_tune(
            models_to_be_evaluated,
            cv_with_pipeline=cv_with_pipeline,
            n_rows=X_train.shape[0],
            n_cols=X_train.shape[1],
            inner_cv_folds=inner_cv_folds,
        )
    else:
        models = classification_models_to_tune(
            models_to_be_evaluated,
            cv_with_pipeline=cv_with_pipeline,
            n_rows=X_train.shape[0],
            n_cols=X_train.shape[1],
            inner_cv_folds=inner_cv_folds,
        )

    return X_train, X_test, pipeline, models


def _cv_workflow_with_pipeline(
    X_train,
    X_test,
    y_train,
    y_test,
    operation,
    pipeline,
    models,
    inner_cv_folds: int,
    inner_cv_rep: int,
    tuning_trials: int,
    scoring,
    average="macro",
    tune_imp_numeric=False,
    tune_scaler=False,
    tune_imp_categorical=False,
):
    # Assert attributes
    if operation == "regression":
        assert (
            scoring in AVAILABLE_SCORES_REGRESSION
        ), string.cv_workflow.assert_score_message + str(AVAILABLE_SCORES_REGRESSION)
    else:
        assert (
            scoring in AVAILABLE_SCORES_CLASSIFICATION
        ), string.cv_workflow.assert_score_message + str(
            AVAILABLE_SCORES_CLASSIFICATION
        )
    assert (
        inner_cv_folds in AVAILABLE_NUMBER_OF_INNER_CV_FOLDS
    ), string.cv_workflow.assert_n_of_inner_cv_folds_message + str(
        AVAILABLE_NUMBER_OF_INNER_CV_FOLDS
    )
    assert inner_cv_rep >= 1, string.cv_workflow.assert_n_of_inner_cv_repetitions
    assert tuning_trials >= 0, string.cv_workflow.assert_n_of_tuning_trials

    # Initiate dictionary with all scores to compute
    scores_to_compute = _compute_dictionary_with_scores_to_compute_in_cv(
        operation, average=average
    )

    # Create inner_cv object
    if operation == "regression":
        inner_cv = RepeatedStratifiedKFoldReg(
            n_splits=inner_cv_folds, n_repeats=inner_cv_rep, random_state=123
        )
    else:
        inner_cv = RepeatedStratifiedKFold(
            n_splits=inner_cv_folds, n_repeats=inner_cv_rep, random_state=123
        )

    collect_scores_cv = pd.DataFrame()
    collect_scores_test = pd.DataFrame()
    collect_models = []
    collect_tuning_studies = []

    for name, p_grid, model in models:
        print("Starting cross-validation of:", name)

        # Cross-validate a baseline model
        start = time()

        pipeline_baseline = deepcopy(pipeline)
        pipeline_baseline.steps.append([name, model])
        cv_scores = cross_validate(
            pipeline_baseline,
            X=X_train,
            y=y_train,
            scoring=scores_to_compute,
            cv=inner_cv,
        )  # error_score="raise"
        stop = time()
        time_cv = stop - start

        cv_scores_df = _convert_scores_cv_dict_to_dataframe(
            name, cv_scores, time_cv, n_tuning_trials=0
        )
        collect_scores_cv = pd.concat([collect_scores_cv, cv_scores_df], axis=0)

        pipeline_baseline.fit(X=X_train, y=y_train)
        test_scores = _compute_scores_test(
            name,
            scores_to_compute,
            pipeline_baseline,
            X_test,
            y_test,
            n_tuning_trials=0,
        )
        collect_scores_test = pd.concat([collect_scores_test, test_scores], axis=0)

        collect_models.append(pipeline_baseline)
        collect_tuning_studies.append(
            string.cv_workflow.tuning_studies_for_baseline_models
        )

        # Cross-validate a tuned model
        if tuning_trials > 0:
            if (
                (name == "Linear_Regression")
                & (tune_imp_numeric is False)
                & (tune_scaler is False)
                & (tune_imp_categorical is False)
            ):
                continue
            start = time()

            pipeline_baseline = deepcopy(pipeline)
            pipeline_baseline.steps.append([name, model])

            # Include data preprocessing procedures in hyperparameter tuning
            p_grid = tune_data_preprocessing(
                params_grid=p_grid,
                tune_imp_numeric=tune_imp_numeric,
                tune_scaler=tune_scaler,
                tune_imp_categorical=tune_imp_categorical,
            )

            optuna_search = OptunaSearchCV(
                estimator=pipeline_baseline,
                param_distributions=p_grid,
                cv=inner_cv,
                n_trials=tuning_trials,
                refit=True,
                scoring=scores_to_compute[scoring],
                random_state=123,
                verbose=0,
            )

            optuna_search.fit(X=X_train, y=y_train)

            pipeline_tuned = optuna_search.best_estimator_

            cv_scores = cross_validate(
                pipeline_tuned,
                X=X_train,
                y=y_train,
                scoring=scores_to_compute,
                cv=inner_cv,
            )
            stop = time()
            time_cv = stop - start

            cv_scores_df = _convert_scores_cv_dict_to_dataframe(
                name, cv_scores, time_cv, n_tuning_trials=tuning_trials
            )
            collect_scores_cv = pd.concat([collect_scores_cv, cv_scores_df], axis=0)

            test_scores = _compute_scores_test(
                name,
                scores_to_compute,
                pipeline_tuned,
                X_test,
                y_test,
                n_tuning_trials=tuning_trials,
            )
            collect_scores_test = pd.concat([collect_scores_test, test_scores], axis=0)

            collect_models.append(pipeline_tuned)
            collect_tuning_studies.append(optuna_search.study_)

    return (
        collect_scores_cv,
        collect_scores_test,
        collect_models,
        collect_tuning_studies,
    )


def _cv_workflow_without_pipeline(
    X_train,
    X_test,
    y_train,
    y_test,
    operation,
    pipeline,
    feature_selection,
    models,
    inner_cv_folds: int,
    inner_cv_rep: int,
    tuning_trials: int,
    scoring,
    average="macro",
):
    # Assert attributes
    if operation == "regression":
        assert (
            scoring in AVAILABLE_SCORES_REGRESSION
        ), string.cv_workflow.assert_score_message + str(AVAILABLE_SCORES_REGRESSION)
    else:
        assert (
            scoring in AVAILABLE_SCORES_CLASSIFICATION
        ), string.cv_workflow.assert_score_message + str(
            AVAILABLE_SCORES_CLASSIFICATION
        )
    assert (
        inner_cv_folds in AVAILABLE_NUMBER_OF_INNER_CV_FOLDS
    ), string.cv_workflow.assert_n_of_inner_cv_folds_message + str(
        AVAILABLE_NUMBER_OF_INNER_CV_FOLDS
    )
    assert inner_cv_rep >= 1, string.cv_workflow.assert_n_of_inner_cv_repetitions
    assert tuning_trials >= 0, string.cv_workflow.assert_n_of_tuning_trials

    # Initiate dictionary with all scores to compute
    scores_to_compute = _compute_dictionary_with_scores_to_compute_in_cv(
        operation, average=average
    )

    # Create inner_cv object
    if operation == "regression":
        inner_cv = RepeatedStratifiedKFoldReg(
            n_splits=inner_cv_folds, n_repeats=inner_cv_rep, random_state=123
        )
    else:
        inner_cv = RepeatedStratifiedKFold(
            n_splits=inner_cv_folds, n_repeats=inner_cv_rep, random_state=123
        )

    collect_scores_cv = pd.DataFrame()
    collect_scores_test = pd.DataFrame()
    collect_models = []
    collect_tuning_studies = []

    for name, p_grid, model in models:
        print("Starting cross-validation of:", name)

        # Cross-validate a baseline model
        start = time()

        # Run pipeline to get 'X_train_prep' and 'X_test_prep'
        pipeline.fit(X_train, y_train)
        X_train_prep = pipeline.transform(X_train)
        X_test_prep = pipeline.transform(X_test)
        # Get labels of all features
        labels = _get_feature_names_after_preprocessing(pipeline, includes_model=False)
        # Convert output to Dataframe and add columns names
        X_train_prep = pd.DataFrame(X_train_prep, columns=labels, index=X_train.index)
        X_test_prep = pd.DataFrame(X_test_prep, columns=labels, index=X_test.index)
        # Instantiate a model to compute baseline scores
        model_baseline = deepcopy(model)

        cv_scores = cross_validate(
            model_baseline,
            X=X_train_prep,
            y=y_train,
            scoring=scores_to_compute,
            cv=inner_cv,
        )  # error_score="raise"
        stop = time()
        time_cv = stop - start

        cv_scores_df = _convert_scores_cv_dict_to_dataframe(
            name, cv_scores, time_cv, n_tuning_trials=0
        )
        collect_scores_cv = pd.concat([collect_scores_cv, cv_scores_df], axis=0)

        model_baseline.fit(X=X_train_prep, y=y_train)
        test_scores = _compute_scores_test(
            name,
            scores_to_compute,
            model_baseline,
            X_test_prep,
            y_test,
            n_tuning_trials=0,
        )
        collect_scores_test = pd.concat([collect_scores_test, test_scores], axis=0)

        # Make a new pipeline with fitted pipeline and fitted baseline model
        pipeline_baseline = make_pipeline(pipeline, model_baseline)
        collect_models.append(pipeline_baseline)

        collect_tuning_studies.append(
            string.cv_workflow.tuning_studies_for_baseline_models
        )

        # Cross-validate a tuned model
        if tuning_trials > 0:
            if name == "Linear_Regression":
                continue
            start = time()

            model_baseline = deepcopy(model)
            optuna_search = OptunaSearchCV(
                estimator=model_baseline,
                param_distributions=p_grid,
                cv=inner_cv,
                n_trials=tuning_trials,
                refit=True,
                scoring=scores_to_compute[scoring],
                random_state=123,
                verbose=0,
            )

            optuna_search.fit(X=X_train_prep, y=y_train)

            model_tuned = optuna_search.best_estimator_

            cv_scores = cross_validate(
                model_tuned,
                X=X_train_prep,
                y=y_train,
                scoring=scores_to_compute,
                cv=inner_cv,
            )  # error_score="raise"
            stop = time()
            time_cv = stop - start

            cv_scores_df = _convert_scores_cv_dict_to_dataframe(
                name, cv_scores, time_cv, n_tuning_trials=tuning_trials
            )
            collect_scores_cv = pd.concat([collect_scores_cv, cv_scores_df], axis=0)

            test_scores = _compute_scores_test(
                name,
                scores_to_compute,
                model_tuned,
                X_test_prep,
                y_test,
                n_tuning_trials=tuning_trials,
            )
            collect_scores_test = pd.concat([collect_scores_test, test_scores], axis=0)

            # Make a new pipeline with fitted pipeline and fitted baseline model
            pipeline_tuned = make_pipeline(pipeline, model_tuned)
            collect_models.append(pipeline_tuned)

            collect_tuning_studies.append(optuna_search.study_)

    return (
        collect_scores_cv,
        collect_scores_test,
        collect_models,
        collect_tuning_studies,
    )


######################################
# Private Methods / Helper functions #
######################################


# For Regression problems: create a binned variable for stratified data split
def _compute_binned_variable(data, target_variable, n_binns=10):
    y_binned = pd.qcut(data[target_variable], n_binns, duplicates="drop")
    return y_binned


def _compute_dictionary_with_scores_to_compute_in_cv(operation, average="macro"):
    if operation == "regression":
        scores_to_compute = {
            "MAE": make_scorer(mean_absolute_error, greater_is_better=False),
            "MSE": make_scorer(mean_squared_error, greater_is_better=False),
            "RMSE": make_scorer(
                mean_squared_error, squared=False, greater_is_better=False
            ),
            "RMSLE": make_scorer(_compute_rmsle, greater_is_better=False),
            "R2": make_scorer(r2_score, greater_is_better=True),
            "MAPE": make_scorer(_compute_mape, greater_is_better=False),
        }
    else:
        scores_to_compute = {
            "Accuracy": make_scorer(accuracy_score, greater_is_better=True),
            "AUC": make_scorer(
                _compute_roc_auc, average=average, greater_is_better=True
            ),
            "Recall": make_scorer(
                recall_score, average=average, zero_division=0, greater_is_better=True
            ),
            "Precision": make_scorer(
                precision_score,
                average=average,
                zero_division=0,
                greater_is_better=True,
            ),
            "F1": make_scorer(
                f1_score, average=average, zero_division=0, greater_is_better=True
            ),
            "Kappa": make_scorer(cohen_kappa_score, greater_is_better=True),
            "MCC": make_scorer(matthews_corrcoef, greater_is_better=True),
        }
    return scores_to_compute


# Compute custom evaluation metrics
def _compute_mape(y, y0):
    mask = y != 0
    return (np.fabs(y - y0) / y)[mask].mean()


def _compute_rmsle(y, y0):
    return np.sqrt(np.mean(np.square(np.log1p(y) - np.log1p(y0))))


def _compute_roc_auc(y, y0, average="weighted"):
    # Create a set of all the unique classes using the actual class list
    unique_class = set(y)
    roc_auc_dict = {}
    roc_auc_list = []
    for per_class in unique_class:
        # Create a list of all the classes except the current class
        other_class = [x for x in unique_class if x != per_class]

        # Mark the current class as 1 and all other classes as 0
        new_actual_class = [0 if x in other_class else 1 for x in y]
        new_pred_class = [0 if x in other_class else 1 for x in y0]

        # Use the sklearn metrics method to calculate the roc_auc_score
        roc_auc = roc_auc_score(new_actual_class, new_pred_class, average=average)
        roc_auc_dict[per_class] = roc_auc
        roc_auc_list.append(roc_auc)

    # Option 1: return the mean of all classes | Option 2: return a dictionary with values of each class

    return mean(roc_auc_list)


def _compute_scores_test(
    model_name, scores_to_compute, fitted_model, X, y_true, n_tuning_trials
):
    # Create a string for the model name, including suffix
    if n_tuning_trials == 0:
        model_name_for_df = model_name + "_baseline"
    else:
        model_name_for_df = model_name + "_tuned"
    # Create an empty dictionary to collect the results and add the adapted model name
    results_dict = {}
    results_dict["model"] = model_name_for_df
    # Compute all scores
    for score_name in scores_to_compute:
        score_function = scores_to_compute[score_name]
        results_dict[score_name] = abs(float(score_function(fitted_model, X, y_true)))
    # Convert dictionary to DataFrame
    results_scores_test = pd.DataFrame.from_dict([results_dict])

    return results_scores_test


def _convert_scores_cv_dict_to_dataframe(
    model_name, scores_cv_dict, time_cv, n_tuning_trials
):
    # Compute length of final DataFrame
    length = len(scores_cv_dict["fit_time"])
    # Create a string for the model name, including suffix
    if n_tuning_trials == 0:
        model_name_for_df = model_name + "_baseline"
    else:
        model_name_for_df = model_name + "_tuned"
    # Convert dictionary to DataFrame and remove the prefix
    df_temp = pd.DataFrame(scores_cv_dict)
    df_temp.columns = df_temp.columns.str.removeprefix("test_")
    # Separate the 'time' columns
    time_column_1 = df_temp.pop("fit_time")
    time_column_2 = df_temp.pop("score_time")
    # Convert all scores to absolute values
    df_temp = df_temp.abs()
    # Create an empty DataFrame (with the requiered number of rows) to append all information
    results_df = pd.DataFrame(index=range(length))
    # Add column with the modified model name
    results_df["model"] = model_name_for_df
    # Add column with the total time
    results_df["time"] = time_column_1 + time_column_2
    results_df["time"] += time_cv / length
    # Add all scores
    results_df = pd.concat([results_df, df_temp], axis=1)

    return results_df


class RepeatedStratifiedKFoldReg:
    def __init__(self, n_splits=10, n_repeats=2, random_state=123):
        self.cvkwargs = dict(
            n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
        )
        self.cv = RepeatedStratifiedKFold(**self.cvkwargs)

    def split(self, X, y, groups=None):
        n_bins = self.cvkwargs["n_splits"]
        y_binned = qcut(y, n_bins, duplicates="drop").factorize()[0]
        return self.cv.split(X, y_binned, groups)

    def get_n_splits(self, X, y, groups=None):
        return self.cv.get_n_splits(X, y, groups)


################
# For debuggin #
################


def main():
    data = pd.read_csv("data/2022-06-19_data_final.csv")
    operation = "regression"  # 'classification | 'regression'
    target_variable = "DEPRES"  # 'GENERO' | 'SITLAB' | 'DEPRES'
    train_size = 0.8
    cv_with_pipeline = True
    inner_cv_folds = 10
    scoring = "RMSE"  # 'Accuracy' | 'RMSE'
    # For splitting training and testing data: Create a variable to stratifiy on
    if operation == "classification":
        stratify_on = data[target_variable]
    elif operation == "regression":
        stratify_on = _compute_binned_variable(
            data=data, target_variable=target_variable, n_binns=10
        )
    else:
        raise ValueError(string.assert_operation_message)
    # Split training and testing data: stratified splitting
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(target_variable, axis="columns"),
        data[target_variable],
        train_size=train_size,
        random_state=123,
        shuffle=True,
        stratify=stratify_on,
    )
    # Reset index
    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    # If classification problem, apply LabelEncoder to 'y'
    if operation == "classification":
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train)
        y_test = label_encoder.transform(y_test)
    # Get NUMERICAL and CATEGORICAL columns
    cols_num = X_train.select_dtypes(include=["float", "int"]).columns.to_list()
    cols_cat = X_train.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.to_list()
    # Data preparation
    pipeline = data_preprocessing(
        cols_num,
        cols_cat,
        imputation_numeric="mean",
        scaler="zscore",
        imputation_categorical="most_frequent",
        one_hot_encoding=True,
    )
    # Add feature selection to pipeline
    pipeline = add_feature_selection_to_pipeline(
        pipeline=pipeline, operation=operation, method="l1_svm"
    )
    # Select and add models
    if operation == "regression":
        models = regression_models_to_tune(
            ["Linear_Regression", "LightGBM"],
            cv_with_pipeline=True,
            n_rows=X_train.shape[0],
            n_cols=X_train.shape[1],
            inner_cv_folds=inner_cv_folds,
        )
    else:
        models = classification_models_to_tune(
            ["GaussianNaiveBayes", "LightGBM"],
            cv_with_pipeline=True,
            n_rows=X_train.shape[0],
            n_cols=X_train.shape[1],
            inner_cv_folds=inner_cv_folds,
        )

    # CV
    if cv_with_pipeline is True:
        results_cv, results_test, results_models = _cv_workflow_with_pipeline(
            X_train,
            X_test,
            y_train,
            y_test,
            operation=operation,
            scoring=scoring,
            pipeline=pipeline,
            models=models,
            inner_cv_folds=inner_cv_folds,
            inner_cv_rep=1,
            tuning_trials=2,
        )

    #    return print(results_df.groupby(by='model').mean()), results_models
    return (
        print(results_cv.groupby(by="model").mean()),
        print(results_test.sort_values(by="model")),
        results_models,
    )


if __name__ == "__main__":
    main()