# Import moduls from local directories
from assets.strings import in_classification_and_regression as string

# Import the required libraries
import numpy as np
import pandas as pd

from boruta import boruta_py

from optuna.distributions import CategoricalDistribution

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import LinearSVC, LinearSVR
from sklearn.linear_model import LassoLarsCV, LogisticRegressionCV
from sklearn.feature_selection import SelectFromModel

from sklearn.base import TransformerMixin

# Import libraries for debugging
from sklearn.model_selection import train_test_split
from modules.utils.preprocessing import (
    data_preprocessing,
    _get_feature_names_after_preprocessing,
)

# 'boruta': (Kumar & Shaikh, 2017; Speiser et al.,2019)
# 'lasso', 'l1_svm': (Sun et al., 2019)

AVAILABLE_FEATURE_SELECTION_METHODS = ("boruta", "lasso_cv", "l1_svm", None)


def add_feature_selection_to_pipeline(pipeline, operation, method):
    assert operation in [
        "classification",
        "regression",
    ], string.feature_selection.assert_operation_message
    assert (
        method in AVAILABLE_FEATURE_SELECTION_METHODS
    ), string.feature_selection.assert_method_message + str(
        AVAILABLE_FEATURE_SELECTION_METHODS
    )

    if method == "boruta":
        pipeline.steps.append(
            ["feature_selection", featureSelectorBoruta(operation=operation)]
        )

    elif (method == "lasso_cv") & (operation == "classification"):
        pipeline.steps.append(["feature_selection", featureSelectorLassoCV_C(cv=5)])

    elif (method == "lasso_cv") & (operation == "regression"):
        pipeline.steps.append(["feature_selection", featureSelectorLassoCV_R(cv=5)])

    elif (method == "l1_svm") & (operation == "classification"):
        pipeline.steps.append(["feature_selection", featureSelectorL1SVM_C(C=1.00)])

    elif (method == "l1_svm") & (operation == "regression"):
        pipeline.steps.append(["feature_selection", featureSelectorL1SVM_R(C=1.00)])

    elif method is None:
        pass

    return pipeline


# This method is not being used, as it is too time-comsuming
def tune_feature_selection(params_grid, operation, tune: bool = False):
    assert operation in [
        "classification",
        "regression",
    ], string.feature_selection.assert_operation_message
    if (tune is True) & (operation == "classification"):
        params_grid["feature_selection"] = CategoricalDistribution(
            choices=[
                featureSelectorBoruta(operation=operation),
                featureSelectorLassoCV_C(cv=5),
                featureSelectorL1SVM_C(C=1.00),
            ]
        )
    if (tune is True) & (operation == "regression"):
        params_grid["feature_selection"] = CategoricalDistribution(
            choices=[
                featureSelectorBoruta(operation=operation),
                featureSelectorLassoCV_R(cv=5),
                featureSelectorL1SVM_R(C=1.00),
            ]
        )
    return params_grid


class featureSelectorBoruta(TransformerMixin):
    """
    Define all parameters and options\n
    operation: 'classification' or 'regression'
    """

    def __init__(self, operation):
        # Assert input values
        assert operation in [
            "classification",
            "regression",
        ], string.feature_selection.assert_operation_message
        # Initiate variables
        self.operation = operation
        super().__init__()

    def __repr__(self):
        string_to_return = "featureSelectorBoruta(n_estimators='auto')"
        return string_to_return

    def fit(self, X, y):
        # BorutaPy accepts numpy arrays only
        X = np.array(X)
        y = np.array(y)
        y = y.ravel()

        if self.operation == "classification":
            # Define random forest classifier
            forest = RandomForestClassifier(
                n_jobs=-1, class_weight="balanced", max_depth=5
            )
            # Instantiate Boruta feature selection method
            self.feature_selector = boruta_py.BorutaPy(
                forest, n_estimators="auto", verbose=0, random_state=123
            )

        else:
            # Define random forest regressor
            forest = RandomForestRegressor(n_jobs=-1, max_depth=5)
            # Instantiate Boruta feature selection method
            self.feature_selector = boruta_py.BorutaPy(
                forest, n_estimators="auto", verbose=0, random_state=123
            )

        self.feature_selector.fit(X=X, y=y)
        return self.feature_selector

    def transform(self, X):
        # BorutaPy accepts numpy arrays only
        X = np.array(X)
        # Call transform() on X to filter it down to selected feature
        X_filtered = self.feature_selector.transform(X)

        return X_filtered

    def get_feature_names_out(self, input_features):
        ranking = pd.DataFrame(self.feature_selector.ranking_)
        indices_of_confirmed_features = ranking[ranking[0] == 1].index
        return input_features[indices_of_confirmed_features]


class featureSelectorLassoCV_C(TransformerMixin):
    def __init__(self, cv: int = 5, random_state: int = 123):
        # Initiate variables
        self.cv = cv
        self.random_state = random_state
        super().__init__()

    def __repr__(self):
        string_to_return = (
            "featureSelectorLassoCV_C(cv="
            + str(self.cv)
            + ", random_state="
            + str(self.random_state)
            + ")"
        )
        return string_to_return

    def fit(self, X, y):
        estimator = LogisticRegressionCV(
            penalty="l1",
            solver="liblinear",
            cv=self.cv,
            dual=False,
            n_jobs=-1,
            random_state=self.random_state,
        ).fit(X, y)
        # Instantiate support vector feature selection method
        self.feature_selector = SelectFromModel(estimator, prefit=True)
        self.feature_selector.fit(X=X, y=y)

        return self.feature_selector

    def transform(self, X):
        # Call transform() on X to filter it down to selected feature
        X_filtered = self.feature_selector.transform(X)

        return X_filtered

    def get_feature_names_out(self, input_features):
        return self.feature_selector.get_feature_names_out(input_features)


class featureSelectorLassoCV_R(TransformerMixin):
    def __init__(self, cv: int = 5):
        # Initiate variables
        self.cv = cv
        super().__init__()

    def __repr__(self):
        string_to_return = "featureSelectorLassoCV_C(cv=" + str(self.cv) + ")"
        return string_to_return

    def fit(self, X, y):
        estimator = LassoLarsCV(cv=self.cv, n_jobs=-1).fit(X, y)
        # Instantiate support vector feature selection method
        self.feature_selector = SelectFromModel(estimator, prefit=True)
        self.feature_selector.fit(X=X, y=y)

        return self.feature_selector

    def transform(self, X):
        # Call transform() on X to filter it down to selected feature
        X_filtered = self.feature_selector.transform(X)

        return X_filtered

    def get_feature_names_out(self, input_features):
        return self.feature_selector.get_feature_names_out(input_features)


class featureSelectorL1SVM_C(TransformerMixin):
    def __init__(self, C: float = 1.00, random_state: int = 123):
        # Initiate variables
        self.C = C
        self.random_state = random_state
        super().__init__()

    def __repr__(self):
        string_to_return = (
            'featureSelectorL1SVM_C(penalty="l1"'
            + ", C="
            + str(self.C)
            + ", random_state="
            + str(self.random_state)
            + ")"
        )
        return string_to_return

    def fit(self, X, y):
        estimator = LinearSVC(
            C=self.C, penalty="l1", dual=False, random_state=self.random_state
        ).fit(X, y)
        # Instantiate support vector feature selection method
        self.feature_selector = SelectFromModel(estimator, prefit=True)
        self.feature_selector.fit(X=X, y=y)

        return self.feature_selector

    def transform(self, X):
        # Call transform() on X to filter it down to selected feature
        X_filtered = self.feature_selector.transform(X)

        return X_filtered

    def get_feature_names_out(self, input_features):
        return self.feature_selector.get_feature_names_out(input_features)


class featureSelectorL1SVM_R(TransformerMixin):
    def __init__(self, C: float = 1.00, random_state: int = 123):
        # Initiate variables
        self.C = C
        self.random_state = random_state
        super().__init__()

    def __repr__(self):
        string_to_return = (
            'featureSelectorL1SVM_R(penalty="l1"'
            + ", C="
            + str(self.C)
            + ", random_state="
            + str(self.random_state)
            + ")"
        )
        return string_to_return

    def fit(self, X, y):
        # Define support vector regressor
        estimator = LinearSVR(
            C=self.C,
            loss="epsilon_insensitive",
            dual=True,
            random_state=self.random_state,
        ).fit(X, y)
        # Instantiate support vector feature selection method
        self.feature_selector = SelectFromModel(estimator, prefit=True)

        self.feature_selector.fit(X=X, y=y)
        return self.feature_selector

    def transform(self, X):
        # Call transform() on X to filter it down to selected feature
        X_filtered = self.feature_selector.transform(X)

        return X_filtered

    def get_feature_names_out(self, input_features):
        return self.feature_selector.get_feature_names_out(input_features)


###############
# For testing #
###############


def main():
    data = pd.read_csv("data/2022-06-19_data_final.csv")
    operation = "regression"  # 'classification | 'regression'
    target_variable = "DEPRES"  # 'GENERO' | 'DEPRES'
    # Split training and testing data: stratified splitting
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(target_variable, axis="columns"),
        data[target_variable],
        train_size=0.8,
        random_state=123,
        shuffle=True,
    )
    # Get NUMERICAL and CATEGORICAL columns
    cols_num = X_train.select_dtypes(include=["float", "int"]).columns.to_list()
    cols_cat = X_train.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.to_list()
    # Data preparation
    pipeline = data_preprocessing(
        cols_num,
        cols_cat,
        imputation_numeric="most_frequent",
        imputation_categorical="most_frequent",
        scaler="zscore",
    )

    # Add feature selection to pipeline
    pipeline = add_feature_selection_to_pipeline(
        pipeline=pipeline, operation=operation, method="lasso"
    )  # 'boruta', 'gini', 'shap', 'lasso', 'l1_svm'

    # Run pipeline
    pipeline.fit(X_train, y_train)

    X_train_prep = pipeline.transform(X_train)
    X_test_prep = pipeline.transform(X_test)

    # Get labels of all features
    labels = _get_feature_names_after_preprocessing(pipeline, includes_model=False)
    # get labels of confirmed features
    labels_confirmed_features = pipeline.named_steps[
        "feature_selection"
    ].get_feature_names_out(labels)
    print(labels_confirmed_features)
    # Convert output to Dataframe and add columns names
    X_train_prep = pd.DataFrame(
        X_train_prep, columns=labels_confirmed_features, index=X_train.index
    )
    X_test_prep = pd.DataFrame(
        X_test_prep, columns=labels_confirmed_features, index=X_test.index
    )

    return print(pd.DataFrame(X_train_prep).info(), pd.DataFrame(X_test_prep).info())


if __name__ == "__main__":
    main()
