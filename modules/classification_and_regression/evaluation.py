# Import moduls from local directories
from assets.colors import get_color
from modules.classification_and_regression.cv_workflow import RepeatedStratifiedKFoldReg
from modules.classification_and_regression.cv_workflow import (
    _compute_dictionary_with_scores_to_compute_in_cv,
)
from modules.utils.plots import _convert_fig_to_html

# Import the required libraries
import matplotlib.pyplot as plt
import numpy as np
from optuna.importance import get_param_importances
from optuna.visualization import plot_optimization_history
from optuna.visualization._utils import COLOR_SCALE
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scikitplot.metrics import plot_cumulative_gain, plot_ks_statistic, plot_lift_curve
from scipy.stats import t
from sklearn.calibration import CalibrationDisplay
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.utils import estimator_html_repr
import streamlit as st
from yellowbrick.model_selection import LearningCurve
from yellowbrick.classifier import ClassificationReport
from yellowbrick.classifier import DiscriminationThreshold, PrecisionRecallCurve, ROCAUC
from yellowbrick.regressor import PredictionError, ResidualsPlot

# Import libraries for debugging
from modules.classification_and_regression.main import k_fold_cross_validation
from modules.utils.load_and_save_data import read_csv


# Compare two models with the "corrected repeated k-fold cv t-test": (Nadeau and Bengio, 2003; Bouckaert and Eibe, 2004)
# https://medium.com/analytics-vidhya/using-the-corrected-paired-students-t-test-for-comparing-the-performance-of-machine-learning-dc6529eaa97f
@st.cache_resource(ttl=3600, max_entries=10)
class corrected_repeated_t_test:
    def __init__(
        self,
        scores_model_1,
        scores_model_2,
        n_folds: int,
        n: int,
    ):
        # Compute the difference between the Generalization Errors of both models
        diff = np.array(scores_model_1) - np.array(scores_model_2)
        # Compute the mean of differences
        d_bar = np.mean(diff)
        # Compute the variance of differences
        sigma2 = np.var(diff)
        # Compute the modified variance: 1/(n_folds-1)==(number of data points used for testing / number of data points used for training) # noqa: E501
        sigma2_mod = sigma2 * (1 / n + 1 / (n_folds - 1))
        # Compute the t_static
        t_static = d_bar / np.sqrt(sigma2_mod)
        # Compute p-value
        p_value = (1 - (t.cdf(x=abs(t_static), df=n - 1))) * 2
        # To calculate effect size (Rosenthal,1994; Howell, 2012)
        cohens_d = abs(2 * t_static) / np.sqrt(n - 1)
        # r_coeficient = np.sqrt((t_static**2) / ((t_static**2) + (n - 1)))
        # DataFrame to output descriptives and statistics
        self.result_descriptives = pd.DataFrame(
            {
                "model": ["model_1", "model_2"],
                "mean": [scores_model_1.mean(), scores_model_2.mean()],
                "sd": [scores_model_1.std(), scores_model_2.std()],
            }
        )
        self.result_statistics = pd.DataFrame(
            [[t_static, n - 1, p_value, cohens_d]],
            columns=["t_statistic", "df", "p_value", "cohens_d"],
        )


#################
# Generic Plots #
#################
def plot_pipeline(pipeline):
    fig_html = estimator_html_repr(pipeline)
    return fig_html


# Shows how the size of training data influences the model to diagnose if a model suffers
# more from variance error vs. bias error.
def plot_learning_curve(
    pipeline,
    data,
    target_variable,
    operation,
    evaluation_score,
    average="macro",
    label_encoder=None,
    cv_folds: int = 10,
    cv_rep: int = 1,
):
    # Drop all rows with NaNs in target_Variable
    data = data.dropna(subset=target_variable)
    # Remove data duplicates while retaining the first one
    data = data.drop_duplicates(keep="first", inplace=False)
    # Separate X and y
    y = pd.Series(data[target_variable])
    X = data.drop(target_variable, axis=1)
    # If classification problem, apply LabelEncoder to 'y'
    if operation == "classification":
        y = label_encoder.transform(y)
    else:
        pass
    # Create cross validation object
    if operation == "regression":
        cv_splits = RepeatedStratifiedKFoldReg(
            n_splits=cv_folds, n_repeats=cv_rep, random_state=123
        )
    else:
        cv_splits = RepeatedStratifiedKFold(
            n_splits=cv_folds, n_repeats=cv_rep, random_state=123
        )
    # Initiate dictionary with all available scores to compute
    scores_to_compute = _compute_dictionary_with_scores_to_compute_in_cv(
        operation, average=average
    )
    # Instantiate visualizer and fit it
    visualizer = LearningCurve(
        pipeline,
        cv=cv_splits,
        scoring=scores_to_compute[evaluation_score],
        n_jobs=-1,
        random_state=123,
    )
    visualizer.fit(X, y)
    visualizer.finalize()
    visualizer.ax.set(ylabel=evaluation_score)
    fig_html = _convert_fig_to_html(visualizer.fig)
    plt.clf()
    return fig_html


########################
# Classification plots #
########################
def plot_classification_calibration_curve(pipeline, X_test, y_test):
    # Instantiate visualizer and fit it
    visualizer = CalibrationDisplay.from_estimator(
        pipeline, X_test, y_test, name=pipeline.steps[-1][0]
    )
    fig_html = _convert_fig_to_html(visualizer.figure_)
    plt.clf()
    return fig_html


def plot_classification_class_prediction(
    pipeline,
    label_encoder,
    X_test,
    y_test,
    color: str = "magma",
    template: str = "gridon",
):
    # Use pipeline to make predictions
    y_pred = pipeline.predict(X_test)
    # Reverse encoding of the target variable
    data_with_y_test_and_y_pred = pd.DataFrame()
    data_with_y_test_and_y_pred["predicted class"] = label_encoder.inverse_transform(
        y_pred
    )
    data_with_y_test_and_y_pred["true class"] = label_encoder.inverse_transform(y_test)

    data_to_be_plotted = pd.DataFrame(
        data_with_y_test_and_y_pred[["true class", "predicted class"]].value_counts()
    ).reset_index()
    data_to_be_plotted.columns = ["true class", "predicted class", "Count"]
    fig_variable = px.bar(
        data_to_be_plotted.sort_values(by="predicted class"),
        x="predicted class",
        y="Count",
        color="true class",
        barmode="stack",
        color_discrete_sequence=get_color(
            color, len(data_with_y_test_and_y_pred["true class"].value_counts())
        ),
        template=template,
    )
    fig_variable.update_layout(xaxis_type="category")
    fig_variable.update_layout(width=600, height=400)
    return fig_variable


def plot_classification_confusion_matrix(
    pipeline, label_encoder, X_test, y_test, color: str = "magma"
):
    from sklearn.metrics import confusion_matrix

    # Use LabelEnconder to get classes
    classes = label_encoder.classes_
    # Instantiate visualizer and fit it
    y_pred = pipeline.predict(X_test)
    confusion_matrix_numpy_array = confusion_matrix(y_test, y_pred)
    fig_variable = px.imshow(
        confusion_matrix_numpy_array,
        x=classes,
        y=classes,
        labels=dict(x="predicted class", y="true class"),
        text_auto=True,
        color_continuous_scale=get_color(
            color, len(pd.DataFrame(confusion_matrix_numpy_array).value_counts())
        ),
        template="simple_white",
    )
    fig_variable.update_layout(xaxis_type="category")
    return fig_variable


def plot_classification_cumulative_gain(pipeline, label_encoder, X_test, y_test):
    # Compute prections
    y_predict = pipeline.predict_proba(X_test)
    # Instantiate visualizer and fit it
    visualizer = plot_cumulative_gain(y_test, y_predict)
    visualizer.legend(labels=label_encoder.classes_)
    fig_html = _convert_fig_to_html(visualizer.figure)
    plt.clf()
    return fig_html


def plot_classification_lift_curve(pipeline, label_encoder, X_test, y_test):
    # Compute prections
    y_predict = pipeline.predict_proba(X_test)
    # Instantiate visualizer and fit it
    visualizer = plot_lift_curve(y_test, y_predict)
    visualizer.legend(labels=label_encoder.classes_)
    fig_html = _convert_fig_to_html(visualizer.figure)
    plt.clf()
    return fig_html


def plot_classification_ks_statistic(pipeline, label_encoder, X_test, y_test):
    # Compute prections
    y_predict = pipeline.predict_proba(X_test)
    # Instantiate visualizer and fit it
    visualizer = plot_ks_statistic(y_test, y_predict)
    visualizer.legend(labels=label_encoder.classes_)
    fig_html = _convert_fig_to_html(visualizer.figure)
    plt.clf()
    return fig_html


def plot_classification_precicion_recall(
    pipeline, label_encoder, X_train, X_test, y_train, y_test, per_class=True
):
    # Use LabelEnconder to get classes
    classes = label_encoder.classes_
    # Instantiate visualizer and fit it
    visualizer = PrecisionRecallCurve(
        pipeline, classes=classes, per_class=True, iso_f1_curves=True, support=True
    )  # per_class=per_class
    visualizer.fit(X_train, y_train)
    visualizer.score(X_test, y_test)
    visualizer.ax.set_xlim([0.0, 1.00])
    visualizer.ax.set_xlim([0.0, 1.05])
    visualizer.finalize()
    fig_html = _convert_fig_to_html(visualizer.fig)
    plt.clf()
    return fig_html


def plot_classification_report(
    pipeline, label_encoder, X_train, X_test, y_train, y_test
):
    # Use LabelEnconder to get classes
    classes = label_encoder.classes_
    # Instantiate visualizer and fit it
    visualizer = ClassificationReport(pipeline, classes=classes, support=True)
    visualizer.fit(X_train, y_train)
    visualizer.score(X_test, y_test)
    visualizer.fig.tight_layout()
    visualizer.finalize()
    fig_html = _convert_fig_to_html(visualizer.fig)
    plt.clf()
    return fig_html


def plot_classification_roc_curves(
    pipeline, label_encoder, X_train, X_test, y_train, y_test
):
    # Use LabelEnconder to get classes
    classes = label_encoder.classes_
    # Instantiate visualizer and fit it
    visualizer = ROCAUC(pipeline, classes=classes, support=True)
    visualizer.fit(X_train, y_train)
    visualizer.score(X_test, y_test)
    visualizer.fig.tight_layout()
    visualizer.finalize()
    fig_html = _convert_fig_to_html(visualizer.fig)
    plt.clf()
    return fig_html


def plot_classification_threshold(
    pipeline, label_encoder, X_train, X_test, y_train, y_test
):
    # Use LabelEnconder to get classes
    classes = label_encoder.classes_
    # Instantiate visualizer and fit it
    visualizer = DiscriminationThreshold(pipeline, classes=classes, support=True)
    visualizer.fit(X_train, y_train)
    visualizer.score(X_test, y_test)
    visualizer.fig.tight_layout()
    visualizer.finalize()
    fig_html = _convert_fig_to_html(visualizer.fig)
    plt.clf()
    return fig_html


####################
# Regression plots #
####################
def plot_regression_prediction_error(pipeline, X_train, X_test, y_train, y_test):
    # Instantiate visualizer and fit it
    visualizer = PredictionError(pipeline)
    visualizer.fit(X_train, y_train)
    visualizer.score(X_test, y_test)
    visualizer.fig.tight_layout()
    visualizer.finalize()
    fig_html = _convert_fig_to_html(visualizer.fig)
    plt.clf()
    return fig_html


def plot_regression_residuals(
    pipeline, X_train, X_test, y_train, y_test, qqplot: bool = False
):
    # Instantiate visualizer and fit it
    if qqplot is False:
        visualizer = ResidualsPlot(pipeline, hist=True)
    else:
        visualizer = ResidualsPlot(pipeline, hist=False, qqplot=True)
    visualizer.fit(X_train, y_train)
    visualizer.score(X_test, y_test)
    visualizer.fig.tight_layout()
    visualizer.finalize()
    fig_html = _convert_fig_to_html(visualizer.fig)
    plt.clf()
    return fig_html


################
# Tuning plots #
################
def get_tuning_trials_dataframe(tuning_study):
    tuning_trials_df = tuning_study.trials_dataframe()
    new_column_names = []
    for value in tuning_trials_df.columns.to_list():
        value = value.removeprefix("params_")
        new_column_names.append(value.removeprefix("user_attrs_"))
    tuning_trials_df.columns = new_column_names
    tuning_trials_df.rename(columns={"number": "trial"}, inplace=True)
    return tuning_trials_df


def plot_tuning_optimization_history(tuning_study):
    if tuning_study is str:
        output = tuning_study
    else:
        output = plot_optimization_history(tuning_study)
    return output


def plot_tuning_param_importances(tuning_study):
    if tuning_study is str:
        fig = tuning_study
    else:
        n_repetitions = 30
        results_to_be_plotted = pd.DataFrame()
        for i in range(n_repetitions):
            np.random.seed(i)
            param_importance_dict = get_param_importances(
                tuning_study, params=list(tuning_study.best_params.keys())
            )
            param_importance_df = pd.DataFrame(param_importance_dict, index=[0])
            results_to_be_plotted = pd.concat(
                [results_to_be_plotted, param_importance_df], axis=0
            )
        results_to_be_plotted = results_to_be_plotted.mean().T.reset_index()
        results_to_be_plotted.columns = ["Hyperparameter", "Importance"]
        fig = px.bar(
            data_frame=results_to_be_plotted.sort_values(
                by="Importance", ascending=True
            ),
            x="Importance",
            y="Hyperparameter",
            orientation="h",
            color="Importance",
            color_discrete_sequence=COLOR_SCALE,
        )
        string_for_x_axis_title = (
            "Mean Importance for Objective Value ("
            + str(n_repetitions)
            + " repetitions)"
        )
        fig.update_yaxes(automargin=True)
        fig.update_layout(xaxis_title=string_for_x_axis_title, yaxis_title=None)
    return fig


def plot_tuning_slice(tuning_study):
    if tuning_study is str:
        fig = tuning_study
    else:
        # get the DataFrame of tuning trials
        tuning_trials_df = get_tuning_trials_dataframe(tuning_study)
        # For not numerical columns, convert all values to string (in preprocessing are methods)
        for column in tuning_trials_df.select_dtypes(
            exclude=["float", "int"]
        ).columns.to_list():
            tuning_trials_df[column] = tuning_trials_df[column].astype(str)
        # Create a list of all hyperparameters to be plotted
        params_list = list(tuning_study.best_params.keys())
        # Define maximum number of columns, instantiate subplots and set titles
        max_columns = 2
        fig = make_subplots(
            rows=int(np.ceil(len(params_list) / max_columns)),
            cols=max_columns,
            subplot_titles=(params_list),
            shared_yaxes=True,
        )
        # Loop through list and add subplots
        n_col = 1
        n_row = 1
        for param in params_list:
            fig.add_trace(
                go.Scatter(
                    x=tuning_trials_df[param],
                    y=tuning_trials_df["value"],
                    mode="markers",
                    marker={
                        "color": tuning_trials_df["trial"],
                        "colorbar": {"title": "Trial"},
                        "colorscale": COLOR_SCALE,
                    },
                ),
                col=n_col,
                row=n_row,
            )
            n_col += 1
            if n_col > max_columns:
                n_col = 1
                n_row += 1
    fig.update_layout(showlegend=False)
    return fig


######################################
# Private Methods / Helper functions #
######################################
class _get_X_and_y_of_nested_cv:
    def __init__(
        self, X, y, outer_cv_object, outer_cv_folds, needed_cv_rep, needed_cv_fold
    ):
        # Compute the needed counter number
        needed_counter_number = ((needed_cv_rep - 1) * outer_cv_folds) + needed_cv_fold
        # Start outer loop for nested cross-validation
        for counter, (train_index, test_index) in enumerate(
            outer_cv_object.split(X, y), start=1
        ):
            # Split training and testing data
            X_train, X_test = X.loc[X.index[train_index]], X.loc[X.index[test_index]]
            y_train, y_test = y.loc[y.index[train_index]], y.loc[y.index[test_index]]
            if needed_counter_number == counter:
                self.X_train = X_train
                self.X_test = X_test
                self.y_train = y_train
                self.y_test = y_test
                break


def main():
    # results_cv = read_csv("../../data/results_cv.csv")

    data = read_csv("../../data/data_c_and_r_with_missings.csv").drop("Loan_ID", axis=1)

    instance_cross_validation = k_fold_cross_validation(
        operation="classification",  # 'classification' | 'regression'
        data=data,
        target_variable="Gender",  # # 'Gender' | 'Dependents'| 'LoanAmount'
        train_size=0.8,
        imputation_numerical="mean",
        imputation_categorical="most_frequent",
        scaler="zscore",
        one_hot_encoding=True,
        feature_selection="l1_svm",  # 'l1_svm' | None
        models_to_be_evaluated=["KNN", "LightGBM"],
        cv_with_pipeline=True,
        inner_cv_folds=5,
        inner_cv_rep=1,
        evaluation_score="Accuracy",  # 'Accuracy' | 'RMSE'
        tuning_trials=10,
        tune_scaler=True,
    )

    figure_to_plot = plot_tuning_param_importances(
        instance_cross_validation.all_results_tuning[3]
    )

    return figure_to_plot.show()


if __name__ == "__main__":
    main()
