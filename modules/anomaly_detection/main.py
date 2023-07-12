# Import moduls from local directories

# Import the required Libraries
import pandas as pd


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
