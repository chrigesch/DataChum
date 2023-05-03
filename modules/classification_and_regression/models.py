# Import moduls from local directories
from assets.strings import in_classification_and_regression as string

# Import the required libraries - classification models
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# Import the required libraries - regression models
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.svm import LinearSVR

from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

# Import the requiered libraries
import numpy as np
from sklearn.utils.extmath import softmax

# Import the required libraries - TensorFlow models
import os
import logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").disabled = True
logging.getLogger("absl").disabled = True

from tensorflow import keras  # noqa: E402

from keras import layers  # noqa: E402
from keras import regularizers  # noqa: E402
from keras.layers import BatchNormalization  # noqa: E402
from keras.layers import Dropout  # noqa: E402
from keras.callbacks import EarlyStopping  # noqa: E402

from scikeras.wrappers import KerasClassifier, KerasRegressor  # noqa: E402

# Import the required libraries - tuning
from optuna.distributions import (  # noqa: E402
    IntDistribution,
    FloatDistribution,
    CategoricalDistribution,
)

AVAILABLE_MODELS_CLASSIFICATION = (
    "GaussianNaiveBayes",
    "LogisticRegression",
    "Ridge",
    "LDA",
    "SVM_L",
    "SVM_R",
    "KNN",
    "DecisionTree",
    "RandomForest",
    "ExtraTrees",
    "CatBoost",
    "LightGBM",
    "XGB",
    "TensorFlow",
)
AVAILABLE_MODELS_REGRESSION = (
    "Bayesian_Ridge",
    "Linear_Regression",
    "SVM_L",
    "SVM_R",
    "KNN",
    "DecisionTree",
    "RandomForest",
    "ExtraTrees",
    "CatBoost",
    "LightGBM",
    "XGB",
    "TensorFlow",
)


def classification_models_to_tune(
    models: list, cv_with_pipeline: bool, n_rows: int, n_cols: int, inner_cv_folds: int
):
    # Assert input values
    for model in models:
        assert (
            model in AVAILABLE_MODELS_CLASSIFICATION
        ), string.models.assert_model_message + str(AVAILABLE_MODELS_CLASSIFICATION)
    # Initiate empty list to collect the selected models
    models_to_tune = []
    # Loop through imput and add selected models
    for model in models:
        if (model == "GaussianNaiveBayes") & (cv_with_pipeline is False):
            models_to_tune.append(
                (
                    "GaussianNaiveBayes",
                    {"var_smoothing": FloatDistribution(1e-12, 1e-0)},
                    GaussianNB(),
                )
            )
        elif (model == "GaussianNaiveBayes") & (cv_with_pipeline is True):
            models_to_tune.append(
                (
                    "GaussianNaiveBayes",
                    {
                        "GaussianNaiveBayes__var_smoothing": FloatDistribution(
                            1e-12, 1e-0
                        )
                    },
                    GaussianNB(),
                )
            )

        if (model == "LogisticRegression") & (cv_with_pipeline is False):
            models_to_tune.append(
                (
                    "LogisticRegression",
                    {
                        "C": FloatDistribution(0.01, 100),
                        "solver": CategoricalDistribution(
                            choices=("lbfgs", "newton-cg")
                        ),
                        "penalty": CategoricalDistribution(choices=("l2", "none")),
                    },
                    LogisticRegression(
                        random_state=123, n_jobs=-1, verbose=0, max_iter=3000
                    ),
                )
            )
        elif (model == "LogisticRegression") & (cv_with_pipeline is True):
            models_to_tune.append(
                (
                    "LogisticRegression",
                    {
                        "LogisticRegression__C": FloatDistribution(0.01, 100),
                        "LogisticRegression__solver": CategoricalDistribution(
                            choices=("lbfgs", "newton-cg")
                        ),
                        "LogisticRegression__penalty": CategoricalDistribution(
                            choices=("l2", "none")
                        ),
                    },
                    LogisticRegression(
                        random_state=123, n_jobs=-1, verbose=0, max_iter=3000
                    ),
                )
            )

        if (model == "Ridge") & (cv_with_pipeline is False):
            models_to_tune.append(
                (
                    "Ridge",
                    {"alpha": FloatDistribution(0.001, 1)},
                    RidgeClassifier_(random_state=123),
                )
            )
        elif (model == "Ridge") & (cv_with_pipeline is True):
            models_to_tune.append(
                (
                    "Ridge",
                    {"Ridge__alpha": FloatDistribution(0.001, 1)},
                    RidgeClassifier_(random_state=123),
                )
            )

        if (model == "LDA") & (cv_with_pipeline is False):
            models_to_tune.append(
                (
                    "LDA",
                    {
                        "solver": CategoricalDistribution(
                            choices=("eigen", "lsqr", "svd")
                        )
                    },
                    LinearDiscriminantAnalysis(),
                )
            )
        elif (model == "LDA") & (cv_with_pipeline is True):
            models_to_tune.append(
                (
                    "LDA",
                    {
                        "LDA__solver": CategoricalDistribution(
                            choices=("eigen", "lsqr", "svd")
                        )
                    },
                    LinearDiscriminantAnalysis(),
                )
            )

        if (model == "SVM_L") & (cv_with_pipeline is False):
            models_to_tune.append(
                (
                    "SVM_L",
                    {"C": FloatDistribution(0.001, 1)},
                    LinearSVC_(random_state=123, max_iter=3000),
                )
            )
        elif (model == "SVM_L") & (cv_with_pipeline is True):
            models_to_tune.append(
                (
                    "SVM_L",
                    {"SVM_L__C": FloatDistribution(0.001, 1)},
                    LinearSVC_(random_state=123, max_iter=3000),
                )
            )

        if (model == "SVM_R") & (cv_with_pipeline is False):
            models_to_tune.append(
                (
                    "SVM_R",
                    {
                        "C": FloatDistribution(0.001, 1),
                        "gamma": FloatDistribution(0.001, 1),
                    },
                    SVC_(kernel="rbf", cache_size=3000),
                )
            )
        if (model == "SVM_R") & (cv_with_pipeline is True):
            models_to_tune.append(
                (
                    "SVM_R",
                    {
                        "SVM_R__C": FloatDistribution(0.001, 1),
                        "SVM_R__gamma": FloatDistribution(0.001, 1),
                    },
                    SVC_(kernel="rbf", cache_size=3000),
                )
            )

        if (model == "KNN") & (cv_with_pipeline is False):
            models_to_tune.append(
                (
                    "KNN",
                    {
                        "n_neighbors": IntDistribution(
                            2, (n_rows / inner_cv_folds) * (inner_cv_folds - 1)
                        ),
                        "p": IntDistribution(1, 2),
                    },
                    KNeighborsClassifier(n_jobs=-1),
                )
            )
        elif (model == "KNN") & (cv_with_pipeline is True):
            models_to_tune.append(
                (
                    "KNN",
                    {
                        "KNN__n_neighbors": IntDistribution(
                            2, (n_rows / inner_cv_folds) * (inner_cv_folds - 1)
                        ),
                        "KNN__p": IntDistribution(1, 2),
                    },
                    KNeighborsClassifier(n_jobs=-1),
                )
            )

        if (model == "DecisionTree") & (cv_with_pipeline is False):
            models_to_tune.append(
                (
                    "DecisionTree",
                    {
                        "max_depth": IntDistribution(1, n_cols),
                        "max_features": IntDistribution(1, n_cols),
                        "min_samples_leaf": FloatDistribution(0.0, 0.5),
                        "min_samples_split": FloatDistribution(0.0, 1.0),
                        "min_weight_fraction_leaf": FloatDistribution(0.0, 0.5),
                    },
                    DecisionTreeClassifier(random_state=123),
                )
            )
        elif (model == "DecisionTree") & (cv_with_pipeline is True):
            models_to_tune.append(
                (
                    "DecisionTree",
                    {
                        "DecisionTree__max_depth": IntDistribution(1, n_cols),
                        "DecisionTree__max_features": IntDistribution(1, n_cols),
                        "DecisionTree__min_samples_leaf": FloatDistribution(0.0, 0.5),
                        "DecisionTree__min_samples_split": FloatDistribution(0.0, 1.0),
                        "DecisionTree__min_weight_fraction_leaf": FloatDistribution(
                            0.0, 0.5
                        ),
                    },
                    DecisionTreeClassifier(random_state=123),
                )
            )

        if (model == "RandomForest") & (cv_with_pipeline is False):
            models_to_tune.append(
                (
                    "RandomForest",
                    {
                        "criterion": CategoricalDistribution(
                            {"entropy", "gini", "log_loss"}
                        ),
                        "max_depth": IntDistribution(1, n_cols),
                        "max_features": IntDistribution(1, n_cols),
                        "min_samples_split": IntDistribution(2, 10),
                        "min_samples_leaf": IntDistribution(1, 4),
                        "n_estimators": IntDistribution(50, 250),
                    },
                    RandomForestClassifier(random_state=123, n_jobs=-1),
                )
            )
        elif (model == "RandomForest") & (cv_with_pipeline is True):
            models_to_tune.append(
                (
                    "RandomForest",
                    {
                        "RandomForest__max_depth": IntDistribution(1, n_cols),
                        "RandomForest__max_features": IntDistribution(1, n_cols),
                        "RandomForest__criterion": CategoricalDistribution(
                            {"entropy", "gini", "log_loss"}
                        ),
                        "RandomForest__min_samples_split": IntDistribution(2, 10),
                        "RandomForest__min_samples_leaf": IntDistribution(1, 4),
                        "RandomForest__n_estimators": IntDistribution(50, 250),
                    },
                    RandomForestClassifier(random_state=123, n_jobs=-1),
                )
            )

        if (model == "ExtraTrees") & (cv_with_pipeline is False):
            models_to_tune.append(
                (
                    "ExtraTrees",
                    {
                        "max_depth": IntDistribution(1, n_cols),
                        "max_features": IntDistribution(1, n_cols),
                        "criterion": CategoricalDistribution(
                            {"entropy", "gini", "log_loss"}
                        ),
                        "min_samples_split": IntDistribution(2, 10),
                        "min_samples_leaf": IntDistribution(1, 4),
                        "n_estimators": IntDistribution(50, 250),
                    },
                    ExtraTreesClassifier(random_state=123, n_jobs=-1),
                )
            )
        elif (model == "ExtraTrees") & (cv_with_pipeline is True):
            models_to_tune.append(
                (
                    "ExtraTrees",
                    {
                        "ExtraTrees__max_depth": IntDistribution(1, n_cols),
                        "ExtraTrees__max_features": IntDistribution(1, n_cols),
                        "ExtraTrees__criterion": CategoricalDistribution(
                            {"entropy", "gini", "log_loss"}
                        ),
                        "ExtraTrees__min_samples_split": IntDistribution(2, 10),
                        "ExtraTrees__min_samples_leaf": IntDistribution(1, 4),
                        "ExtraTrees__n_estimators": IntDistribution(50, 250),
                    },
                    ExtraTreesClassifier(random_state=123, n_jobs=-1),
                )
            )

        if (model == "CatBoost") & (cv_with_pipeline is False):
            models_to_tune.append(
                (
                    "CatBoost",
                    {
                        "learning_rate": FloatDistribution(1e-5, 1e-1),
                        "n_estimators": IntDistribution(50, 250),
                        "depth": IntDistribution(1, 10),
                        "l2_leaf_reg": IntDistribution(2, 10),
                        "random_strength": IntDistribution(0, 10),
                        "subsample": FloatDistribution(0.5, 1.0),
                    },
                    CatBoostClassifier(
                        random_state=123, allow_writing_files=False, verbose=0
                    ),
                )
            )
        elif (model == "CatBoost") & (cv_with_pipeline is True):
            models_to_tune.append(
                (
                    "CatBoost",
                    {
                        "CatBoost__learning_rate": FloatDistribution(1e-5, 1e-1),
                        "CatBoost__n_estimators": IntDistribution(50, 250),
                        "CatBoost__depth": IntDistribution(1, 10),
                        "CatBoost__l2_leaf_reg": IntDistribution(2, 10),
                        "CatBoost__random_strength": IntDistribution(0, 10),
                        "CatBoost__subsample": FloatDistribution(0.5, 1.0),
                    },
                    CatBoostClassifier(
                        random_state=123, allow_writing_files=False, verbose=0
                    ),
                )
            )

        if (model == "LightGBM") & (cv_with_pipeline is False):
            models_to_tune.append(
                (
                    "LightGBM",
                    {
                        "learning_rate": FloatDistribution(1e-5, 1e-1),
                        "max_depth": IntDistribution(1, 30),
                        "n_estimators": IntDistribution(50, 250),
                        "num_leaves": IntDistribution(20, 300),
                    },
                    LGBMClassifier(
                        boosting_type="gbdt", n_jobs=-1, random_state=123, verbose=-1
                    ),
                )
            )
        elif (model == "LightGBM") & (cv_with_pipeline is True):
            models_to_tune.append(
                (
                    "LightGBM",
                    {
                        "LightGBM__learning_rate": FloatDistribution(1e-5, 1e-1),
                        "LightGBM__max_depth": IntDistribution(1, 30),
                        "LightGBM__n_estimators": IntDistribution(50, 250),
                        "LightGBM__num_leaves": IntDistribution(2, 300),
                    },
                    LGBMClassifier(
                        boosting_type="gbdt", n_jobs=-1, random_state=123, verbose=-1
                    ),
                )
            )

        if (model == "XGB") & (cv_with_pipeline is False):
            models_to_tune.append(
                (
                    "XGB",
                    {
                        "colsample_bytree": FloatDistribution(0.0, 1.0),
                        "num_rounds": IntDistribution(1, 4000),
                        "learning_rate": FloatDistribution(1e-5, 1e-1),
                        "max_depth": IntDistribution(3, 12),
                        "min_child_weight": IntDistribution(1, 100),
                        "n_estimators": IntDistribution(50, 250),
                        "subsample": FloatDistribution(0.5, 1.0),
                    },
                    XGBClassifier(random_state=123, n_jobs=-1, verbosity=0),
                )
            )
        elif (model == "XGB") & (cv_with_pipeline is True):
            models_to_tune.append(
                (
                    "XGB",
                    {
                        "XGB__colsample_bytree": FloatDistribution(0.0, 1.0),
                        "XGB__num_rounds": IntDistribution(1, 4000),
                        "XGB__learning_rate": FloatDistribution(1e-5, 1e-1),
                        "XGB__max_depth": IntDistribution(3, 12),
                        "XGB__min_child_weight": IntDistribution(1, 100),
                        "XGB__n_estimators": IntDistribution(50, 250),
                        "XGB__subsample": FloatDistribution(0.5, 1.0),
                    },
                    XGBClassifier(random_state=123, n_jobs=-1, verbosity=0),
                )
            )

        if (model == "TensorFlow") & (cv_with_pipeline is False):
            models_to_tune.append(
                (
                    "TensorFlow",
                    {
                        "model__neuronsDenseLayer1": IntDistribution(1, n_cols**2),
                        "model__neuronsDenseLayer2": IntDistribution(
                            1, (n_cols**2) / 2
                        ),
                        "model__dropout1": FloatDistribution(0.0, 0.6),
                        "model__dropout2": FloatDistribution(0.0, 0.6),
                        "model__wrl2_1": FloatDistribution(0.0, 0.005),
                        "model__wrl2_2": FloatDistribution(0.0, 0.005),
                        "model__activation": CategoricalDistribution(
                            {
                                "elu",
                                "exponential",
                                "LeakyReLU",
                                "relu",
                                "selu",
                                "sigmoid",
                                "softplus",
                                "softsign",
                                "tanh",
                            }
                        ),
                        "optimizer": CategoricalDistribution(
                            {
                                "Adam",
                                "Adadelta",
                                "Adagrad",
                                "Adamax",
                                "Ftrl",
                                "Nadam",
                                "RMSprop",
                                "SGD",
                            }
                        ),
                        "optimizer__learning_rate": FloatDistribution(1e-5, 1e-1),
                        "batch_size": CategoricalDistribution({32, 64, 128}),
                    },
                    KerasClassifier(
                        get_tf_classifier,
                        loss="sparse_categorical_crossentropy",
                        epochs=50,
                        verbose=0,
                        random_state=123,
                        callbacks=[
                            EarlyStopping(
                                monitor="loss", patience=3, verbose=0, min_delta=1e-4
                            )
                        ],
                    ),
                )
            )
        elif (model == "TensorFlow") & (cv_with_pipeline is True):
            models_to_tune.append(
                (
                    "TensorFlow",
                    {
                        "TensorFlow__model__neuronsDenseLayer1": IntDistribution(
                            1, n_cols**2
                        ),
                        "TensorFlow__model__neuronsDenseLayer2": IntDistribution(
                            1, (n_cols**2) / 2
                        ),
                        "TensorFlow__model__dropout1": FloatDistribution(0.0, 0.6),
                        "TensorFlow__model__dropout2": FloatDistribution(0.0, 0.6),
                        "TensorFlow__model__wrl2_1": FloatDistribution(0.0, 0.005),
                        "TensorFlow__model__wrl2_2": FloatDistribution(0.0, 0.005),
                        "TensorFlow__model__activation": CategoricalDistribution(
                            {
                                "elu",
                                "exponential",
                                "LeakyReLU",
                                "relu",
                                "selu",
                                "sigmoid",
                                "softplus",
                                "softsign",
                                "tanh",
                            }
                        ),
                        "TensorFlow__optimizer": CategoricalDistribution(
                            {
                                "Adam",
                                "Adadelta",
                                "Adagrad",
                                "Adamax",
                                "Ftrl",
                                "Nadam",
                                "RMSprop",
                                "SGD",
                            }
                        ),
                        "TensorFlow__optimizer__learning_rate": FloatDistribution(
                            1e-5, 1e-1
                        ),
                        "TensorFlow__batch_size": CategoricalDistribution(
                            {32, 64, 128}
                        ),
                    },
                    KerasClassifier(
                        get_tf_classifier,
                        loss="sparse_categorical_crossentropy",
                        epochs=50,
                        verbose=0,
                        random_state=123,
                        callbacks=[
                            EarlyStopping(
                                monitor="loss", patience=3, verbose=0, min_delta=1e-4
                            )
                        ],
                    ),
                )
            )

    return models_to_tune


def regression_models_to_tune(
    models: list, cv_with_pipeline: bool, n_rows: int, n_cols: int, inner_cv_folds: int
):
    # Assert input values
    for model in models:
        assert (
            model in AVAILABLE_MODELS_REGRESSION
        ), string.models.assert_model_message + str(AVAILABLE_MODELS_REGRESSION)
    # Initiate empty list to collect the selected models
    models_to_tune = []
    # Loop through imput and add selected models
    for model in models:
        if (model == "Bayesian_Ridge") & (cv_with_pipeline is False):
            models_to_tune.append(
                (
                    "Bayesian_Ridge",
                    {
                        "alpha_1": FloatDistribution(1e-5, 1e-0),
                        "alpha_2": FloatDistribution(1e-5, 1e-0),
                        "lambda_1": FloatDistribution(1e-5, 1e-0),
                        "lambda_2": FloatDistribution(1e-5, 1e-0),
                    },
                    BayesianRidge(),
                )
            )
        elif (model == "Bayesian_Ridge") & (cv_with_pipeline is True):
            models_to_tune.append(
                (
                    "Bayesian_Ridge",
                    {
                        "Bayesian_Ridge__alpha_1": FloatDistribution(1e-5, 1e-0),
                        "Bayesian_Ridge__alpha_2": FloatDistribution(1e-5, 1e-0),
                        "Bayesian_Ridge__lambda_1": FloatDistribution(1e-5, 1e-0),
                        "Bayesian_Ridge__lambda_2": FloatDistribution(1e-5, 1e-0),
                    },
                    BayesianRidge(),
                )
            )

        if model == "Linear_Regression":
            models_to_tune.append(("Linear_Regression", {}, LinearRegression()))

        if (model == "KNN") & (cv_with_pipeline is False):
            models_to_tune.append(
                (
                    "KNN",
                    {
                        "n_neighbors": IntDistribution(
                            2, (n_rows / inner_cv_folds) * (inner_cv_folds - 1)
                        ),
                        "p": IntDistribution(1, 2),
                    },
                    KNeighborsRegressor(),
                )
            )
        elif (model == "KNN") & (cv_with_pipeline is True):
            models_to_tune.append(
                (
                    "KNN",
                    {
                        "KNN__n_neighbors": IntDistribution(
                            2, (n_rows / inner_cv_folds) * (inner_cv_folds - 1)
                        ),
                        "KNN__p": IntDistribution(1, 2),
                    },
                    KNeighborsRegressor(),
                )
            )

        if (model == "SVM_L") & (cv_with_pipeline is False):
            models_to_tune.append(
                (
                    "SVM_L",
                    {
                        "C": FloatDistribution(0.01, 1),
                        "epsilon": FloatDistribution(0.01, 1),
                    },
                    LinearSVR(random_state=123, max_iter=3000),
                )
            )
        elif (model == "SVM_L") & (cv_with_pipeline is True):
            models_to_tune.append(
                (
                    "SVM_L",
                    {
                        "SVM_L__C": FloatDistribution(0.01, 1),
                        "SVM_L__epsilon": FloatDistribution(0.01, 1),
                    },
                    LinearSVR(random_state=123, max_iter=3000),
                )
            )

        if (model == "SVM_R") & (cv_with_pipeline is False):
            models_to_tune.append(
                (
                    "SVM_R",
                    {
                        "C": FloatDistribution(0.001, 1),
                        "gamma": FloatDistribution(0.001, 1),
                    },
                    SVR(kernel="rbf", cache_size=3000),
                )
            )
        elif (model == "SVM_R") & (cv_with_pipeline is True):
            models_to_tune.append(
                (
                    "SVM_R",
                    {
                        "SVM_R__C": FloatDistribution(0.001, 1),
                        "SVM_R__gamma": FloatDistribution(0.001, 1),
                    },
                    SVR(kernel="rbf", cache_size=3000),
                )
            )

        if (model == "DecisionTree") & (cv_with_pipeline is False):
            models_to_tune.append(
                (
                    "DecisionTree",
                    {
                        "max_depth": IntDistribution(1, n_cols),
                        "max_features": IntDistribution(1, n_cols),
                        "min_samples_leaf": FloatDistribution(0.0, 0.5),
                        "min_samples_split": FloatDistribution(0.0, 1.0),
                        "min_weight_fraction_leaf": FloatDistribution(0.0, 0.5),
                    },
                    DecisionTreeRegressor(random_state=123),
                )
            )
        elif (model == "DecisionTree") & (cv_with_pipeline is True):
            models_to_tune.append(
                (
                    "DecisionTree",
                    {
                        "DecisionTree__max_depth": IntDistribution(1, n_cols),
                        "DecisionTree__max_features": IntDistribution(1, n_cols),
                        "DecisionTree__min_samples_leaf": FloatDistribution(0.0, 0.5),
                        "DecisionTree__min_samples_split": FloatDistribution(0.0, 1.0),
                        "DecisionTree__min_weight_fraction_leaf": FloatDistribution(
                            0.0, 0.5
                        ),
                    },
                    DecisionTreeRegressor(random_state=123),
                )
            )

        if (model == "RandomForest") & (cv_with_pipeline is False):
            models_to_tune.append(
                (
                    "RandomForest",
                    {
                        "max_depth": IntDistribution(1, n_cols),
                        "max_features": IntDistribution(1, n_cols),
                        "min_samples_leaf": FloatDistribution(0.0, 0.5),
                        "n_estimators": IntDistribution(50, 250),
                    },
                    RandomForestRegressor(random_state=123, n_jobs=-1),
                )
            )
        elif (model == "RandomForest") & (cv_with_pipeline is True):
            models_to_tune.append(
                (
                    "RandomForest",
                    {
                        "RandomForest__max_depth": IntDistribution(1, n_cols),
                        "RandomForest__max_features": IntDistribution(1, n_cols),
                        "RandomForest__min_samples_leaf": FloatDistribution(0.0, 0.5),
                        "RandomForest__n_estimators": IntDistribution(50, 250),
                    },
                    RandomForestRegressor(random_state=123, n_jobs=-1),
                )
            )

        if (model == "ExtraTrees") & (cv_with_pipeline is False):
            models_to_tune.append(
                (
                    "ExtraTrees",
                    {
                        "max_depth": IntDistribution(1, n_cols),
                        "max_features": IntDistribution(1, n_cols),
                        "min_samples_leaf": FloatDistribution(0.0, 0.5),
                        "n_estimators": IntDistribution(50, 250),
                    },
                    ExtraTreesRegressor(random_state=123, n_jobs=-1),
                )
            )
        elif (model == "ExtraTrees") & (cv_with_pipeline is True):
            models_to_tune.append(
                (
                    "ExtraTrees",
                    {
                        "ExtraTrees__max_depth": IntDistribution(1, n_cols),
                        "ExtraTrees__max_features": IntDistribution(1, n_cols),
                        "ExtraTrees__min_samples_leaf": FloatDistribution(0.0, 0.5),
                        "ExtraTrees__n_estimators": IntDistribution(50, 250),
                    },
                    ExtraTreesRegressor(random_state=123, n_jobs=-1),
                )
            )

        if (model == "CatBoost") & (cv_with_pipeline is False):
            models_to_tune.append(
                (
                    "CatBoost",
                    {
                        "learning_rate": FloatDistribution(1e-5, 1e-1),
                        "depth": IntDistribution(1, 10),
                        "l2_leaf_reg": IntDistribution(2, 10),
                        "random_strength": IntDistribution(0, 10),
                        "n_estimators": IntDistribution(50, 250),
                        "subsample": FloatDistribution(0.5, 1.0),
                    },
                    CatBoostRegressor(
                        random_state=123, allow_writing_files=False, verbose=0
                    ),
                )
            )
        elif (model == "CatBoost") & (cv_with_pipeline is True):
            models_to_tune.append(
                (
                    "CatBoost",
                    {
                        "CatBoost__learning_rate": FloatDistribution(1e-5, 1e-1),
                        "CatBoost__depth": IntDistribution(1, 10),
                        "CatBoost__l2_leaf_reg": IntDistribution(2, 10),
                        "CatBoost__random_strength": IntDistribution(0, 10),
                        "CatBoost__n_estimators": IntDistribution(50, 250),
                        "CatBoost__subsample": FloatDistribution(0.5, 1.0),
                    },
                    CatBoostRegressor(
                        random_state=123, allow_writing_files=False, verbose=0
                    ),
                )
            )

        if (model == "LightGBM") & (cv_with_pipeline is False):
            models_to_tune.append(
                (
                    "LightGBM",
                    {
                        "learning_rate": FloatDistribution(1e-5, 1e-1),
                        "max_depth": IntDistribution(1, 30),
                        "n_estimators": IntDistribution(50, 250),
                        "num_leaves": IntDistribution(20, 300),
                    },
                    LGBMRegressor(
                        boosting_type="gbdt", n_jobs=-1, random_state=123, verbose=-1
                    ),
                )
            )
        elif (model == "LightGBM") & (cv_with_pipeline is True):
            models_to_tune.append(
                (
                    "LightGBM",
                    {
                        "LightGBM__learning_rate": FloatDistribution(1e-5, 1e-1),
                        "LightGBM__max_depth": IntDistribution(1, 30),
                        "LightGBM__n_estimators": IntDistribution(50, 250),
                        "LightGBM__num_leaves": IntDistribution(20, 300),
                    },
                    LGBMRegressor(
                        boosting_type="gbdt", n_jobs=-1, random_state=123, verbose=-1
                    ),
                )
            )

        if (model == "XGB") & (cv_with_pipeline is False):
            models_to_tune.append(
                (
                    "XGB",
                    {
                        "colsample_bytree": FloatDistribution(0.0, 1.0),
                        "num_rounds": IntDistribution(1, 4000),
                        "learning_rate": FloatDistribution(1e-5, 1e-1),
                        "max_depth": IntDistribution(1, 12),
                        "min_child_weight": IntDistribution(1, 100),
                        "n_estimators": IntDistribution(50, 250),
                        "subsample": FloatDistribution(0.5, 1.0),
                    },
                    XGBRegressor(random_state=123, n_jobs=-1),
                )
            )
        elif (model == "XGB") & (cv_with_pipeline is True):
            models_to_tune.append(
                (
                    "XGB",
                    {
                        "XGB__colsample_bytree": FloatDistribution(0.0, 1.0),
                        "XGB__num_rounds": IntDistribution(1, 4000),
                        "XGB__learning_rate": FloatDistribution(1e-5, 1e-1),
                        "XGB__max_depth": IntDistribution(1, 12),
                        "XGB__min_child_weight": IntDistribution(1, 100),
                        "XGB__n_estimators": IntDistribution(50, 250),
                        "XGB__subsample": FloatDistribution(0.5, 1.0),
                    },
                    XGBRegressor(random_state=123, n_jobs=-1),
                )
            )

        if (model == "TensorFlow") & (cv_with_pipeline is False):
            models_to_tune.append(
                (
                    "TensorFlow",
                    {
                        "model__neuronsDenseLayer1": IntDistribution(1, n_cols**2),
                        "model__neuronsDenseLayer2": IntDistribution(
                            1, (n_cols**2) / 2
                        ),
                        "model__dropout1": FloatDistribution(0.0, 0.6),
                        "model__dropout2": FloatDistribution(0.0, 0.6),
                        "model__wrl2_1": FloatDistribution(0.0, 0.005),
                        "model__wrl2_2": FloatDistribution(0.0, 0.005),
                        "model__activation": CategoricalDistribution(
                            {
                                "elu",
                                "exponential",
                                "LeakyReLU",
                                "relu",
                                "selu",
                                "sigmoid",
                                "softplus",
                                "softsign",
                                "tanh",
                            }
                        ),
                        "optimizer": CategoricalDistribution(
                            {
                                "Adam",
                                "Adadelta",
                                "Adagrad",
                                "Adamax",
                                "Ftrl",
                                "Nadam",
                                "RMSprop",
                                "SGD",
                            }
                        ),
                        "optimizer__learning_rate": FloatDistribution(1e-5, 1e-1),
                        "batch_size": CategoricalDistribution({32, 64, 128}),
                    },
                    KerasRegressor(
                        get_tf_regressor,
                        loss="mse",
                        epochs=50,
                        verbose=0,
                        random_state=123,
                        callbacks=[
                            EarlyStopping(
                                monitor="loss", patience=3, verbose=0, min_delta=1e-4
                            )
                        ],
                    ),
                )
            )
        elif (model == "TensorFlow") & (cv_with_pipeline is True):
            models_to_tune.append(
                (
                    "TensorFlow",
                    {
                        "TensorFlow__model__neuronsDenseLayer1": IntDistribution(
                            1, n_cols**2
                        ),
                        "TensorFlow__model__neuronsDenseLayer2": IntDistribution(
                            1, (n_cols**2) / 2
                        ),
                        "TensorFlow__model__dropout1": FloatDistribution(0.0, 0.6),
                        "TensorFlow__model__dropout2": FloatDistribution(0.0, 0.6),
                        "TensorFlow__model__wrl2_1": FloatDistribution(0.0, 0.005),
                        "TensorFlow__model__wrl2_2": FloatDistribution(0.0, 0.005),
                        "TensorFlow__model__activation": CategoricalDistribution(
                            {
                                "elu",
                                "exponential",
                                "LeakyReLU",
                                "relu",
                                "selu",
                                "sigmoid",
                                "softplus",
                                "softsign",
                                "tanh",
                            }
                        ),
                        "TensorFlow__optimizer": CategoricalDistribution(
                            {
                                "Adam",
                                "Adadelta",
                                "Adagrad",
                                "Adamax",
                                "Ftrl",
                                "Nadam",
                                "RMSprop",
                                "SGD",
                            }
                        ),
                        "TensorFlow__optimizer__learning_rate": FloatDistribution(
                            1e-5, 1e-1
                        ),
                        "TensorFlow__batch_size": CategoricalDistribution(
                            {32, 64, 128}
                        ),
                    },
                    KerasRegressor(
                        get_tf_regressor,
                        loss="mse",
                        epochs=50,
                        verbose=0,
                        random_state=123,
                        callbacks=[
                            EarlyStopping(
                                monitor="loss", patience=3, verbose=0, min_delta=1e-4
                            )
                        ],
                    ),
                )
            )

    return models_to_tune


# Add predict_proba method to those Classifier that do not include it
class LinearSVC_(LinearSVC):
    def predict_proba(self, X):
        decision = self.decision_function(X)
        if decision.ndim == 1:
            decision_2d = np.c_[-decision, decision]
        else:
            decision_2d = decision
        return softmax(decision_2d)


class RidgeClassifier_(RidgeClassifier):
    def predict_proba(self, X):
        decision = self.decision_function(X)
        if decision.ndim == 1:
            decision_2d = np.c_[-decision, decision]
        else:
            decision_2d = decision
        return softmax(decision_2d)


class SVC_(SVC):
    def predict_proba(self, X):
        decision = self.decision_function(X)
        if decision.ndim == 1:
            decision_2d = np.c_[-decision, decision]
        else:
            decision_2d = decision
        return softmax(decision_2d)


# Define architecture of the Neural Network
def get_tf_classifier(
    meta,
    neuronsDenseLayer1=64,
    neuronsDenseLayer2=32,
    dropout1=0.0,
    dropout2=0.0,
    wrl2_1=0.0,
    wrl2_2=0.0,
    activation="relu",
):
    # meta is a special argument that will be handed a dict containing input metadata
    n_features_in_ = meta["n_features_in_"]
    n_classes_ = meta["n_classes_"]

    # Initialize a sequential model and add layers etc.
    model = keras.Sequential()
    model.add(keras.Input(shape=n_features_in_))
    model.add(BatchNormalization())
    model.add(
        layers.Dense(
            units=neuronsDenseLayer1,
            kernel_regularizer=regularizers.L2(wrl2_1),
            activation=activation,
        )
    )
    model.add(Dropout(dropout1))
    model.add(
        layers.Dense(
            units=neuronsDenseLayer2,
            kernel_regularizer=regularizers.L2(wrl2_2),
            activation=activation,
        )
    )
    model.add(Dropout(dropout2))
    model.add(layers.Dense(units=n_classes_, activation="softmax"))

    return model


# Define architecture of the Neural Network
def get_tf_regressor(
    meta,
    neuronsDenseLayer1=64,
    neuronsDenseLayer2=32,
    dropout1=0.0,
    dropout2=0.0,
    wrl2_1=0.0,
    wrl2_2=0.0,
    activation="relu",
):
    # meta is a special argument that will be handed a dict containing input metadata
    n_features_in_ = meta["n_features_in_"]

    # Initialize a sequential model and add layers etc.
    model = keras.Sequential()
    model.add(keras.Input(shape=n_features_in_))
    model.add(BatchNormalization())
    model.add(
        layers.Dense(
            units=neuronsDenseLayer1,
            kernel_regularizer=regularizers.L2(wrl2_1),
            activation=activation,
        )
    )
    model.add(Dropout(dropout1))
    model.add(
        layers.Dense(
            units=neuronsDenseLayer2,
            kernel_regularizer=regularizers.L2(wrl2_2),
            activation=activation,
        )
    )
    model.add(Dropout(dropout2))
    model.add(layers.Dense(units=1, activation=activation))

    return model


#################
# For debugging #
#################


def main():
    models_classification = classification_models_to_tune(
        ["GaussianNaiveBayes"],
        cv_with_pipeline=True,
        n_rows=100,
        n_cols=5,
        inner_cv_folds=10,
    )

    models_regression = regression_models_to_tune(
        ["Bayesian_Ridge"],
        cv_with_pipeline=True,
        n_rows=100,
        n_cols=5,
        inner_cv_folds=10,
    )

    return print(models_classification), print(models_regression)


if __name__ == "__main__":
    main()
