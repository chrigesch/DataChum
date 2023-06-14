# Import moduls from local directory

# Import the required libraries
import numpy as np
import pandas as pd
import miceforest as mf
from optuna.distributions import CategoricalDistribution
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split

from sklearn.experimental import enable_iterative_imputer  # noqa F401
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.base import (
    TransformerMixin,
)  # to create a custom miceForestImputter and missForestImputter

from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    MaxAbsScaler,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)
from sklearn.compose import ColumnTransformer

AVAILABLE_IMPUTATION_NUMERICAL = (
    "mean",
    "median",
    "most_frequent",
    "mice_forest",
    "miss_forest",
    None,
)
AVAILABLE_SCALER = ("maxabs", "minmax", "robust", "yeo_johnson", "zscore", None)
AVAILABLE_IMPUTATION_CATEGORICAL = ("most_frequent", "mice_forest", "miss_forest", None)
AVAILABLE_ONE_HOT_ENCODER = (True, False)


def data_preprocessing(
    cols_num,
    cols_cat,
    imputation_numerical="mean",
    scaler="zscore",
    imputation_categorical="most_frequent",
    one_hot_encoding=True,
):
    # Assert input values
    assert imputation_numerical in AVAILABLE_IMPUTATION_NUMERICAL, (
        "Unrecognized value, 'imputation_numerical' should be one of the following: "
        + str(AVAILABLE_IMPUTATION_NUMERICAL)
    )
    assert (
        scaler in AVAILABLE_SCALER
    ), "Unrecognized value, 'scaler' should be one of the following: " + str(
        AVAILABLE_SCALER
    )
    assert imputation_categorical in AVAILABLE_IMPUTATION_CATEGORICAL, (
        "Unrecognized value, 'imputation_categorical' should be one of the following: "
        + str(AVAILABLE_IMPUTATION_CATEGORICAL)
    )
    assert (
        one_hot_encoding in AVAILABLE_ONE_HOT_ENCODER
    ), "Unrecognized value, 'one_hot_encoding' should be one of the following: " + str(
        AVAILABLE_ONE_HOT_ENCODER
    )

    # Instantiate pipeline for NUMERICAL data
    transformer_num = Pipeline(steps=[])

    # Add imputation method for NUMERICAL features
    if imputation_numerical in ["mean", "median", "most_frequent"]:
        transformer_num.steps.append(
            ("imput_num", SimpleImputer(strategy=imputation_numerical))
        )
    elif imputation_numerical == "mice_forest":
        transformer_num.steps.append(("imput_num", miceForestImputer()))
    elif imputation_numerical == "miss_forest":
        transformer_num.steps.append(
            (
                "imput_num",
                IterativeImputer(
                    estimator=RandomForestRegressor(
                        n_estimators=4,
                        max_depth=10,
                        bootstrap=True,
                        max_samples=0.5,
                        n_jobs=-1,
                        random_state=123,
                    ),
                    max_iter=20,
                ),
            )
        )
    elif imputation_numerical is None:
        pass
    else:
        raise ValueError()

    # Add scaler method for NUMERICAL features
    if scaler == "maxabs":
        transformer_num.steps.append(("scale", MaxAbsScaler()))
    elif scaler == "minmax":
        transformer_num.steps.append(("scale", MinMaxScaler()))
    elif scaler == "robust":
        transformer_num.steps.append(("scale", RobustScaler()))
    elif scaler == "yeo_johnson":
        transformer_num.steps.append(("scale", PowerTransformer()))
    elif scaler == "zscore":
        transformer_num.steps.append(("scale", StandardScaler()))
    elif scaler is None:
        pass
    else:
        raise ValueError()

    # Instantiate pipeline for CATEGORICAL data
    transformer_cat = Pipeline(steps=[])

    # Add imputation method for CATEGORICAL features
    if imputation_categorical == "most_frequent":
        transformer_cat.steps.append(
            ("imput_cat", SimpleImputer(strategy=imputation_categorical))
        )
    elif imputation_categorical == "mice_forest":
        transformer_cat.steps.append(("imput_cat", miceForestImputer()))
    elif imputation_categorical == "miss_forest":
        transformer_cat.steps.append(("imput_cat", missForestClassifierImputer()))
    elif imputation_categorical is None:
        pass
    else:
        raise ValueError(
            "Unrecognized value, 'imputation_categorical' should be one of the following: "
        )

    # Add one hot encoder
    if one_hot_encoding is True:
        transformer_cat.steps.append(
            ("one_hot", OneHotEncoder(handle_unknown="ignore", drop="if_binary"))
        )

    # Instantiate an empty ColumnTransformer
    preprocessor = ColumnTransformer(transformers=[], remainder="passthrough")

    # Add pipelines of NUMERICAL and CATEGORICAL data to the ColumnTransformer
    preprocessor.transformers.append(("prep_num", transformer_num, cols_num))
    preprocessor.transformers.append(("prep_cat", transformer_cat, cols_cat))

    # Instantiate an empty pipeline for modelling
    pipeline_for_cross_validation = Pipeline(steps=[])
    # Add preprocesor to pipeline
    pipeline_for_cross_validation.steps.append(["preprocessor", preprocessor])

    return pipeline_for_cross_validation


def tune_data_preprocessing(
    params_grid,
    tune_imp_numerical: bool = False,
    tune_scaler: bool = False,
    tune_imp_categorical: bool = False,
):
    if tune_imp_numerical is True:
        params_grid["preprocessor__prep_num__imput_num"] = CategoricalDistribution(
            choices=[
                SimpleImputer(strategy="mean"),
                SimpleImputer(strategy="median"),
                SimpleImputer(strategy="most_frequent"),
                miceForestImputer(),
                IterativeImputer(
                    estimator=RandomForestRegressor(
                        n_estimators=4,
                        bootstrap=True,
                        max_samples=0.5,
                        n_jobs=-1,
                        random_state=123,
                    ),
                    max_iter=20,
                ),
            ]
        )
    if tune_scaler is True:
        params_grid["preprocessor__prep_num__scale"] = CategoricalDistribution(
            choices=[
                MaxAbsScaler(),
                MinMaxScaler(),
                RobustScaler(),
                PowerTransformer(),
                StandardScaler(),
            ]
        )
    if tune_imp_categorical is True:
        params_grid["preprocessor__prep_cat__imput_cat"] = CategoricalDistribution(
            choices=[
                SimpleImputer(strategy="most_frequent"),
                miceForestImputer(),
                missForestClassifierImputer(),
            ]
        )
    return params_grid


class miceForestImputer(TransformerMixin):
    def __init__(self, iterations=2, n_estimators=50, random_state=123):
        self.iterations = iterations
        self.n_estimators = n_estimators
        self.random_state = random_state
        super().__init__()

    def __repr__(self):
        string_to_return = (
            "miceForestImputer(iterations="
            + str(self.iterations)
            + ", n_estimators="
            + str(self.n_estimators)
            + ", random_state="
            + str(self.random_state)
            + ")"
        )
        return string_to_return

    def fit(self, X, y=None):
        data_mice_forest_f = X.copy(deep=True)
        # Convert all 'object' to 'category' (mice_forest & miss_forest)
        data_mice_forest_f = _convert_from_dtype_to_dtype(
            data_mice_forest_f, from_dtype="object", to_dtype="category"
        )
        # If data does not contain any missing value, create ONE random NAN
        if np.array(data_mice_forest_f.isna()).sum() == 0:
            data_mice_forest_f.iloc[0, 0] = np.NAN
        # Instantiate kernel
        self.imputter_micef = mf.ImputationKernel(
            data_mice_forest_f,
            datasets=1,
            save_all_iterations=False,
            variable_schema=None,
            train_nonmissing=True,
            random_state=self.random_state,
        )
        # Run the MICE algorithm for 5 iterations
        self.imputter_micef.mice(
            iterations=self.iterations, n_estimators=self.n_estimators
        )

        return self

    def transform(self, X):
        data_mice_forest_t = X.copy(deep=True)
        # Convert all 'object' to 'category' (mice_forest & miss_forest)
        data_mice_forest_t = _convert_from_dtype_to_dtype(
            data_mice_forest_t, from_dtype="object", to_dtype="category"
        )
        # If data contains a missing value
        if np.array(data_mice_forest_t.isna()).sum() > 0:
            # Impute with the fitted kernel and return the completed data
            self.imputter_micef.compile_candidate_preds()
            data_mice_forest_t_completed = self.imputter_micef.transform(
                data_mice_forest_t
            )
        else:
            data_mice_forest_t_completed = data_mice_forest_t.copy(deep=True)
        # Convert all 'category' to 'object' (mice_forest & miss_forest)
        data_mice_forest_t_completed = _convert_from_dtype_to_dtype(
            data_mice_forest_t_completed, from_dtype="category", to_dtype="object"
        )

        return data_mice_forest_t_completed


class missForestClassifierImputer(TransformerMixin):
    def __init__(
        self,
        max_iter: int = 20,
        n_estimators: int = 4,
        max_depth: int = 10,
        bootstrap: bool = True,
        max_samples: float = 0.5,
        n_jobs=-1,
        random_state: int = 123,
    ):
        self.max_iter = max_iter
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.bootstrap = bootstrap
        self.max_samples = max_samples
        self.n_jobs = n_jobs
        self.random_state = random_state
        super().__init__()

    def __repr__(self):
        string_to_return = (
            "missForestImputer(max_iter="
            + str(self.max_iter)
            + ", n_estimators="
            + str(self.n_estimators)
            + ", max_depth="
            + str(self.max_depth)
            + ", bootstrap="
            + str(self.bootstrap)
            + ", max_samples="
            + str(self.max_samples)
            + ", n_jobs="
            + str(self.n_jobs)
            + ", random_state="
            + str(self.random_state)
            + ")"
        )
        return string_to_return

    def fit(self, X, y=None):
        # Instantiate and fit Ordinal Encoder
        self.ordinal_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=np.nan
        )
        data_encoded = self.ordinal_encoder.fit_transform(X)
        # Instantiate and fit the imputer
        self.imputer = IterativeImputer(
            estimator=RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                bootstrap=self.bootstrap,
                max_samples=self.max_samples,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            ),
            max_iter=self.max_iter,
        )
        self.imputer = self.imputer.fit(data_encoded)

        return self

    def transform(self, X):
        # Transform X with the already fitted encoder
        data_encoded = self.ordinal_encoder.transform(X)
        # Imput missings with the already fitted imputer
        data_imputed = self.imputer.transform(data_encoded)
        data_decoded = self.ordinal_encoder.inverse_transform(data_imputed)

        return data_decoded


######################################
# Private Methods / Helper functions #
######################################


def _convert_from_dtype_to_dtype(data, from_dtype, to_dtype):
    data_result = data.copy(deep=True)
    object_columns = data_result.select_dtypes([from_dtype]).columns.tolist()
    for column in object_columns:
        data_result[column] = data_result[column].astype(to_dtype)
    return data_result


def _fit_pipeline_to_get_preprocessed_data(
    cols_num, cols_cat, pipeline, X_train, X_test
):
    # Fit pipeline
    pipeline.fit(X_train)
    # Transform
    X_train_prep = pipeline.transform(X_train)
    X_test_prep = pipeline.transform(X_test)
    # If there are categorical features, include their labels
    if len(cols_cat) > 0:
        encoded_cat = (
            pipeline.named_steps["preprocessor"]
            .named_transformers_["prep_cat"]
            .named_steps["one_hot"]
            .get_feature_names_out(cols_cat)
        )
        labels = np.concatenate([cols_num, encoded_cat])
    else:
        labels = cols_num
    # Change columns names ([LightGBM] Do not support special JSON characters in feature name.)
    import re

    labels_new = [re.sub(r"[^A-Za-z0-9_]+", "", value) for value in labels]
    # Convert output to Dataframe and add columns names
    X_train_prep = pd.DataFrame(X_train_prep, columns=labels_new, index=X_train.index)
    X_test_prep = pd.DataFrame(X_test_prep, columns=labels_new, index=X_test.index)

    return X_train_prep, X_test_prep


def _get_feature_names_after_preprocessing(pipeline, includes_model: bool):
    # Get column names of the pipeline and remove prefixes
    # (do not include the model, which is the last part of the pipeline)
    import re

    col_names_without_prefix = []
    if includes_model is True:
        for element in pipeline[:-1].get_feature_names_out():
            element = element.removeprefix("prep_cat__")
            element = element.removeprefix("prep_num__")
            # Change columns names ([LightGBM] Do not support special JSON characters in feature name.)
            element = re.sub(r"[^A-Za-z0-9_]+", "", element)
            col_names_without_prefix.append(element)
    else:
        for element in pipeline.get_feature_names_out():
            element = element.removeprefix("prep_cat__")
            element = element.removeprefix("prep_num__")
            # Change columns names ([LightGBM] Do not support special JSON characters in feature name.)
            element = re.sub(r"[^A-Za-z0-9_]+", "", element)
            col_names_without_prefix.append(element)
    return col_names_without_prefix


###############
# For testing #
###############


def main():
    data = pd.read_csv("data/data_c_and_r_with_missings.csv").drop("Loan_ID", axis=1)
    target_variable = "ApplicantIncome"
    #    data = pd.read_csv('data/2022-06-19_data_final.csv')
    #    target_variable = 'DEPRES'
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
        imputation_numerical="most_frequent",  # 'mean' | 'miss_forest' | 'most_frequent'
        imputation_categorical="miss_forest",
        scaler="zscore",
    )
    # Fit and transform data
    X_train_prep, X_test_prep = _fit_pipeline_to_get_preprocessed_data(
        cols_num, cols_cat, pipeline, X_train, X_test
    )
    # Get labels of all features
    labels = _get_feature_names_after_preprocessing(pipeline, includes_model=False)

    return print(X_train_prep.info(), X_test_prep.info(), labels)


if __name__ == "__main__":
    main()
