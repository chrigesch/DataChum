# Import the required libraries
from statistics import mean
import numpy as np
import pandas as pd
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

AVAILABLE_SCORES_REGRESSION = ("MAE", "MSE", "RMSE", "RMSLE", "R2", "MAPE")

AVAILABLE_SCORES_CLASSIFICATION = (
    "Accuracy",
    "AUC",
    "Recall",
    "Precision",
    "F1",
    "Kappa",
    "MCC",
)


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
