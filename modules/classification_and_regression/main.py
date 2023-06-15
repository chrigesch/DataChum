# Import moduls from local directories
from modules.classification_and_regression.cv_workflow import _compute_binned_variable
from modules.classification_and_regression.cv_workflow import RepeatedStratifiedKFoldReg
from modules.classification_and_regression.cv_workflow import _cv_preparation
from modules.classification_and_regression.cv_workflow import _cv_workflow_with_pipeline
from modules.classification_and_regression.cv_workflow import (
    _cv_workflow_without_pipeline,
)
from modules.utils.load_and_save_data import read_csv
from modules.utils.preprocessing import clean_strings_and_feature_names

# Import the required Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder


# @st.cache_resource(ttl=3600, max_entries=10)
class k_fold_cross_validation:
    """
    Parameters:
    -----------\n
    operation: str
        'classification' or 'regression'
    data: iterable
        cleaned DataFrame (check if all of the data types are correctly identified)
    target_variable: str
        name of the target column in dataset
    train_size: float
        float between 0.5 and 0.95
    imputation_numerical: str
        'mean', 'median', 'most_frequent', 'mice_forest', 'miss_forest', None
    imputation_categorical: str
        'most_frequent', 'mice_forest', 'miss_forest', None
    scaler: str
        'maxabs' (Scale each feature by its maximum absolute value.), 'minmax' (Scale each feature between zero and
        one.), 'robust' (Scale data according to the quantile range [IQR].), 'yeo-johnson', 'zscore'
    one_hot_encoding: bool
        True, False
    feature_selection: str
        'boruta','lasso_cv','l1_svm', None
    models_to_be_evaluated: str
        if operation=='classification': 'GaussianNaiveBayes','LogisticRegression','Ridge',
        'LinearDiscriminantAnalysis','SVM_L','SVM_R', 'KNN','DecisionTree','RandomForest','ExtraTrees','CatBoost',
        'LightGBM','XGB','TensorFlow' | if operation=='regression': 'Bayesian_Ridge','Linear_Regression','SVM_L',
        'SVM_R','KNN','DecisionTree','RandomForest','ExtraTrees','CatBoost','LightGBM','XGB','TensorFlow'
    cv_with_pipeline: bool
        True, False (pipeline is usefull to avoid overfitting and for deployment
        but might be slower to train)
    inner_cv_folds: int
        integer between 5 and 10
    inner_cv_rep: int
        integer >= 1
    tuning_trials: int
        integer >= 0
    evaluation_score: str
        Strategy to evaluate the performance of the cross-validated model on the test set.
    average: str
        Average used to compute AUC, Recall, Precision and F1.
    tune_imp_numerical: bool
        use hyperparameter tuning to find the best numerical imputer. (will be ignored when cv_with_pipeline=False)
    tune_scaler: bool
        use hyperparameter tuning to find the best scaler. (will be ignored when cv_with_pipeline=False)
    tune_impu_categorical: bool
        use hyperparameter tuning to find the best categorical imputer. (will be ignored when cv_with_pipeline=False)
    """

    def __init__(
        self,
        operation,
        data,
        target_variable: str,
        train_size: float,
        imputation_numerical,
        imputation_categorical,
        scaler,
        one_hot_encoding,
        feature_selection,
        models_to_be_evaluated,
        cv_with_pipeline,
        inner_cv_folds,
        inner_cv_rep,
        tuning_trials,
        evaluation_score,
        average="macro",
        tune_imp_numerical=False,
        tune_scaler=False,
        tune_imp_categorical=False,
    ):
        # Assert input values
        assert (
            0.5 <= train_size <= 0.95
        ), "Unrecognized value, 'train_size' should be between 0.5 and 0.95"
        if tune_imp_categorical is True:
            assert (
                imputation_categorical is not None
            ), "If tune_imp_categorical=True, imputation_categorical must not be None"
        if tune_imp_numerical is True:
            assert (
                imputation_numerical is not None
            ), "If tune_imp_numerical=True, imputation_numerical must not be None"
        if tune_scaler is True:
            assert scaler is not None, "If tune_scaler=True, scaler must not be None"
        # Constructs all the necessary attributes for the instance
        self.procedure = "k_fold"
        self.operation = operation
        self.data = data
        self.target_variable = target_variable
        self.train_size = train_size
        self.imputation_numerical = imputation_numerical
        self.imputation_categorical = imputation_categorical
        self.scaler = scaler
        self.one_hot_encoding = one_hot_encoding
        self.feature_selection = feature_selection
        self.models_to_be_evaluated = models_to_be_evaluated
        self.cv_with_pipeline = cv_with_pipeline
        self.inner_cv_folds = inner_cv_folds
        self.inner_cv_rep = inner_cv_rep
        self.tuning_trials = tuning_trials
        self.evaluation_score = evaluation_score
        self.average = average
        self.tune_imp_numerical = tune_imp_numerical
        self.tune_scaler = tune_scaler
        self.tune_imp_categorical = tune_imp_categorical

        # Drop all DATETIME columns
        cols_tim = self.data.select_dtypes(include=["datetime"]).columns.to_list()
        self.data = self.data.drop(cols_tim, axis=1)
        # Drop all rows with NaNs in target_Variable
        self.data = self.data.dropna(subset=self.target_variable)
        # Remove data duplicates while retaining the first one
        self.data = self.data.drop_duplicates(keep="first", inplace=False)
        # Clean strings and feature names
        self.data = clean_strings_and_feature_names(self.data)
        # If classification problem, apply LabelEncoder to 'y'
        if self.operation == "classification":
            self.label_encoder = LabelEncoder()
            self.data[self.target_variable] = self.label_encoder.fit_transform(
                self.data[self.target_variable]
            )
        # For splitting training and testing data: Create a variable to stratifiy on
        if self.operation == "regression":
            _stratify_on = _compute_binned_variable(
                data=self.data, target_variable=self.target_variable, n_binns=10
            )
        else:
            _stratify_on = self.data[self.target_variable]
        # Split training and testing data: stratified splitting
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.data.drop(self.target_variable, axis="columns"),
            self.data[self.target_variable],
            train_size=self.train_size,
            random_state=123,
            shuffle=True,
            stratify=_stratify_on,
        )
        # Perform preparation for the cross validation: pipeline (preprossessing + feature_selection) & models
        self.X_train, self.X_test, _pipeline, _models = _cv_preparation(
            self.X_train,
            self.X_test,
            self.operation,
            self.inner_cv_folds,
            self.imputation_numerical,
            self.imputation_categorical,
            self.scaler,
            self.one_hot_encoding,
            self.feature_selection,
            self.models_to_be_evaluated,
            self.cv_with_pipeline,
        )
        # Perform cross validation
        if self.cv_with_pipeline is True:
            (
                self.all_results_cv,
                self.all_results_test,
                self.all_results_models,
                self.all_results_tuning,
            ) = _cv_workflow_with_pipeline(
                self.X_train,
                self.X_test,
                self.y_train,
                self.y_test,
                operation=self.operation,
                scoring=self.evaluation_score,
                average=self.average,
                pipeline=_pipeline,
                models=_models,
                inner_cv_folds=self.inner_cv_folds,
                inner_cv_rep=self.inner_cv_rep,
                tuning_trials=self.tuning_trials,
                tune_imp_numerical=self.tune_imp_numerical,
                tune_scaler=self.tune_scaler,
                tune_imp_categorical=self.tune_imp_categorical,
            )

        else:
            (
                self.all_results_cv,
                self.all_results_test,
                self.all_results_models,
                self.all_results_tuning,
            ) = _cv_workflow_without_pipeline(
                self.X_train,
                self.X_test,
                self.y_train,
                self.y_test,
                operation=self.operation,
                scoring=self.evaluation_score,
                average=self.average,
                pipeline=_pipeline,
                feature_selection=self.feature_selection,
                models=_models,
                inner_cv_folds=self.inner_cv_folds,
                inner_cv_rep=self.inner_cv_rep,
                tuning_trials=self.tuning_trials,
            )


AVAILABLE_NUMBER_OF_OUTER_CV_FOLDS = list(range(5, 11))


# @st.cache_resource(ttl=3600, max_entries=10)
class nested_k_fold_cross_validation:
    """
    Parameters:
    -----------\n
    operation: str
        'classification' or 'regression'
    data: iterable
        cleaned DataFrame (check if all of the data types are correctly identified)
    target_variable: str
        name of the target column in dataset.
    imputation_numerical: str
        'mean', 'median', 'most_frequent', 'mice_forest', 'miss_forest', None.
    imputation_categorical: str
        'most_frequent', 'mice_forest', 'miss_forest', None
    scaler: str
        'maxabs' (Scale each feature by its maximum absolute value.), 'minmax' (Scale each feature between
        zero and one.), 'robust' (Scale data according to the quantile range [IQR].), 'yeo-johnson', 'zscore'
    one_hot_encoding: bool
        True, False
    feature_selection: str
        'boruta','lasso_cv','l1_svm', None
    models_to_be_evaluated: str
        if operation=='classification': 'GaussianNaiveBayes','LogisticRegression','Ridge',
        'LinearDiscriminantAnalysis','SVM_L','SVM_R', 'KNN','DecisionTree','RandomForest','ExtraTrees','CatBoost',
        'LightGBM','XGB','TensorFlow' | if operation=='regression': 'Bayesian_Ridge','Linear_Regression','SVM_L',
        'SVM_R', 'KNN','DecisionTree','RandomForest','ExtraTrees','CatBoost','LightGBM','XGB','TensorFlow'
    cv_with_pipeline: bool
        True, False (pipeline is usefull to avoid overfitting and for deployment but might be slower to train)
    outer_cv_folds: int
        integer between 5 and 10
    outer_cv_rep: int
        integer >= 1
    inner_cv_folds: int
        integer between 5 and 10
    inner_cv_rep: int
        integer >= 1
    tuning_trials: int
        integer >= 0
    evaluation_score: str
        Strategy to evaluate the performance of the cross-validated model on the test set.
    average: str
        Average used to compute AUC, Recall, Precision and F1.
    tune_imp_numerical: bool
        use hyperparameter tuning to find the best numerical imputer. (will be ignored when cv_with_pipeline=False)
    tune_scaler: bool
        use hyperparameter tuning to find the best scaler. (will be ignored when cv_with_pipeline=False)
    tune_impu_categorical: bool
        use hyperparameter tuning to find the best categorical imputer. (will be ignored when cv_with_pipeline=False)
    """

    def __init__(
        self,
        operation,
        data,
        target_variable: str,
        imputation_numerical,
        imputation_categorical,
        scaler,
        one_hot_encoding,
        feature_selection,
        models_to_be_evaluated,
        cv_with_pipeline,
        outer_cv_folds: int,
        outer_cv_rep: int,
        inner_cv_folds,
        inner_cv_rep,
        tuning_trials,
        evaluation_score,
        average="macro",
        tune_imp_numerical=False,
        tune_scaler=False,
        tune_imp_categorical=False,
    ):
        # Assert input values
        if tune_imp_categorical is True:
            assert (
                imputation_categorical is not None
            ), "If tune_imp_categorical=True, imputation_categorical must not be None"
        if tune_imp_numerical is True:
            assert (
                imputation_numerical is not None
            ), "If tune_imp_numerical=True, imputation_numerical must not be None"
        if tune_scaler is True:
            assert scaler is not None, "If tune_scaler=True, scaler must not be None"
        assert outer_cv_folds in AVAILABLE_NUMBER_OF_OUTER_CV_FOLDS, (
            "Unrecognized value, 'outer_cv_folds' should be one of the following: "
            + str(AVAILABLE_NUMBER_OF_OUTER_CV_FOLDS)
        )
        assert outer_cv_rep >= 1, "Unrecognized value, 'outer_cv_rep' should be >= 1"
        # Constructs all the necessary attributes for the instance
        self.procedure = "nested"
        self.operation = operation
        self.data = data
        self.target_variable = target_variable
        self.imputation_numerical = imputation_numerical
        self.imputation_categorical = imputation_categorical
        self.scaler = scaler
        self.one_hot_encoding = one_hot_encoding
        self.feature_selection = feature_selection
        self.models_to_be_evaluated = models_to_be_evaluated
        self.cv_with_pipeline = cv_with_pipeline
        self.outer_cv_folds = outer_cv_folds
        self.outer_cv_rep = outer_cv_rep
        self.inner_cv_folds = inner_cv_folds
        self.inner_cv_rep = inner_cv_rep
        self.tuning_trials = tuning_trials
        self.evaluation_score = evaluation_score
        self.average = average
        self.tune_imp_numerical = tune_imp_numerical
        self.tune_scaler = tune_scaler
        self.tune_imp_categorical = tune_imp_categorical

        # Drop all DATETIME columns
        cols_tim = self.data.select_dtypes(include=["datetime"]).columns.to_list()
        self.data = self.data.drop(cols_tim, axis=1)
        # Drop all rows with NaNs in target_Variable
        self.data = self.data.dropna(subset=self.target_variable)
        # Remove data duplicates while retaining the first one
        self.data = self.data.drop_duplicates(keep="first", inplace=False)
        # Clean strings and feature names
        self.data = clean_strings_and_feature_names(self.data)
        # If classification problem, apply LabelEncoder to 'y'
        if self.operation == "classification":
            self.label_encoder = LabelEncoder()
            self.data[self.target_variable] = self.label_encoder.fit_transform(
                self.data[self.target_variable]
            )
        # Separate y from X and convert target variable to numbers
        self.y = pd.Series(self.data[self.target_variable])
        self.X = self.data.drop(self.target_variable, axis=1)

        # Iniziate DataFrames to collect all results
        self.all_results_cv = pd.DataFrame()
        self.all_results_test = pd.DataFrame()
        self.all_results_models = []
        self.all_results_tuning = []

        # Create outer_cv object
        if self.operation == "regression":
            self.outer_cv_object = RepeatedStratifiedKFoldReg(
                n_splits=self.outer_cv_folds,
                n_repeats=self.outer_cv_rep,
                random_state=123,
            )
        else:
            self.outer_cv_object = RepeatedStratifiedKFold(
                n_splits=self.outer_cv_folds,
                n_repeats=self.outer_cv_rep,
                random_state=123,
            )

        # Start outer loop for nested cross-validation
        for counter, (train_index, test_index) in enumerate(
            self.outer_cv_object.split(self.X, self.y), start=1
        ):
            # Split training and testing data
            _X_train, _X_test = (
                self.X.loc[self.X.index[train_index]],
                self.X.loc[self.X.index[test_index]],
            )
            _y_train, _y_test = (
                self.y.loc[self.y.index[train_index]],
                self.y.loc[self.y.index[test_index]],
            )

            # Perform preparation for the cross validation: pipeline (preprossessing + feature_selection) & models
            _X_train, _X_test, _pipeline, _models = _cv_preparation(
                _X_train,
                _X_test,
                self.operation,
                self.inner_cv_folds,
                self.imputation_numerical,
                self.imputation_categorical,
                self.scaler,
                self.one_hot_encoding,
                self.feature_selection,
                self.models_to_be_evaluated,
                self.cv_with_pipeline,
            )
            # Perform cross validation
            if self.cv_with_pipeline is True:
                (
                    _results_cv,
                    _results_test,
                    _results_models,
                    _results_tuning,
                ) = _cv_workflow_with_pipeline(
                    _X_train,
                    _X_test,
                    _y_train,
                    _y_test,
                    operation=self.operation,
                    scoring=self.evaluation_score,
                    average=self.average,
                    pipeline=_pipeline,
                    models=_models,
                    inner_cv_folds=self.inner_cv_folds,
                    inner_cv_rep=self.inner_cv_rep,
                    tuning_trials=self.tuning_trials,
                    tune_imp_numerical=self.tune_imp_numerical,
                    tune_scaler=self.tune_scaler,
                    tune_imp_categorical=self.tune_imp_categorical,
                )

            else:
                (
                    _results_cv,
                    _results_test,
                    _results_models,
                    _results_tuning,
                ) = _cv_workflow_without_pipeline(
                    _X_train,
                    _X_test,
                    _y_train,
                    _y_test,
                    operation=self.operation,
                    scoring=self.evaluation_score,
                    average=self.average,
                    pipeline=_pipeline,
                    feature_selection=self.feature_selection,
                    models=_models,
                    inner_cv_folds=self.inner_cv_folds,
                    inner_cv_rep=self.inner_cv_rep,
                    tuning_trials=self.tuning_trials,
                )

            # Use counter to compute repetition and folder number
            _out_cv_rep = int((counter - 1) / outer_cv_folds + 1)
            _out_cv_fold = counter - (_out_cv_rep - 1) * outer_cv_folds
            print(
                "Finished outer cv repetition",
                _out_cv_rep,
                "and outer cv fold",
                _out_cv_fold,
            )
            # Add counter to DataFrames
            _results_cv["rep"] = _out_cv_rep
            _results_cv["fold"] = _out_cv_fold
            _results_test["rep"] = _out_cv_rep
            _results_test["fold"] = _out_cv_fold

            # Append all results to DataFrames and list (FITTED models are appended)
            self.all_results_cv = pd.concat([self.all_results_cv, _results_cv], axis=0)
            self.all_results_test = pd.concat(
                [self.all_results_test, _results_test], axis=0
            )
            # To avoid nested list, loop through it and append
            for model_to_append in _results_models:
                self.all_results_models.append(model_to_append)
            # To avoid nested list, loop through it and append
            for tuning_study_to_append in _results_tuning:
                self.all_results_tuning.append(tuning_study_to_append)

        self.all_results_cv.reset_index(inplace=True, drop=True)
        self.all_results_test.reset_index(inplace=True, drop=True)


#################
# For debugging #
#################


def main():
    data = read_csv("data/data_c_and_r_with_missings.csv").drop("Loan_ID", axis=1)

    cv_method = "k_fold"  # 'k_fold', 'nested_k_fold'

    if cv_method == "k_fold":
        instance_cross_validation = k_fold_cross_validation(
            operation="regression",  # 'classification' | 'regression'
            data=data,
            target_variable="LoanAmount",  # # 'Gender' | 'Dependents'| 'LoanAmount'
            train_size=0.8,
            imputation_numerical="mean",
            imputation_categorical="most_frequent",  # 'mean'
            scaler="zscore",
            one_hot_encoding=True,
            feature_selection="l1_svm",  # 'l1_svm' | None
            models_to_be_evaluated=["KNN", "LightGBM"],
            cv_with_pipeline=True,
            inner_cv_folds=5,
            inner_cv_rep=1,
            evaluation_score="RMSE",  # 'Accuracy' | 'RMSE'
            tuning_trials=2,
        )

    else:
        instance_cross_validation = nested_k_fold_cross_validation(
            operation="classification",  # 'classification' | 'regression'
            data=data,
            target_variable="Dependents",
            imputation_numerical="mean",
            imputation_categorical="most_frequent",
            scaler="zscore",
            one_hot_encoding=True,
            feature_selection="l1_svm",
            models_to_be_evaluated=[
                "KNN",
                "LightGBM",
            ],  # 'Linear_Regression','KNN,'CatBoost','LightGBM'
            cv_with_pipeline=False,
            outer_cv_folds=5,
            outer_cv_rep=1,
            inner_cv_folds=5,
            inner_cv_rep=1,
            evaluation_score="Accuracy",
            tuning_trials=2,
        )

    return print(
        instance_cross_validation.all_results_cv.groupby(by="model").mean()
    ), print(instance_cross_validation.all_results_test)


if __name__ == "__main__":
    main()
