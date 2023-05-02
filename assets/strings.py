######################
# Strings in modules #
######################


class in_classification_and_regression:
    class cv_workflow:
        assert_operation_message = (
            "Unrecognized value, 'operation' should be one of the following: "
        )
        assert_score_message = (
            "Unrecognized value, 'score' should be one of the following: "
        )
        assert_n_of_inner_cv_folds_message = (
            "Unrecognized value, 'inner_cv_folds' should be one of the following: "
        )
        assert_n_of_inner_cv_repetitions = (
            "Unrecognized value, 'inner_cv_rep' should be >= 1"
        )
        assert_n_of_tuning_trials = "Unrecognized value, 'tuning_trials' should be >= 0"
        tuning_studies_for_baseline_models = "This model has not been tuned"

    class feature_selection:
        assert_method_message = (
            "Unrecognized value, 'method' should be one of the following: "
        )
        assert_operation_message = (
            "Unrecognized value, 'operation' should be one of the following: "
        )

    class interpretation:
        assert_estimation_method_message = (
            "Unrecognized value, 'estimation_method' should be one of the following: "
        )

    class main:
        assert_train_size_message = (
            "Unrecognized value, 'train_size' should be between 0.5 and 0.95"
        )
        assert_n_of_outer_cv_folds_message = (
            "Unrecognized value, 'outer_cv_folds' should be one of the following: "
        )
        assert_n_of_outer_cv_repetitions = (
            "Unrecognized value, 'outer_cv_rep' should be >= 1"
        )
        assert_tune_imp_categorical = (
            "If tune_imp_categorical=True, imputation_categorical must not be None"
        )
        assert_tune_imp_numeric = (
            "If tune_imp_numeric=True, imputation_numeric must not be None"
        )
        assert_tune_scaler = "If tune_scaler=True, scaler must not be None"

    class models:
        assert_model_message = (
            "Unrecognized value, model should be one of the following: "
        )
        assert_operation_message = (
            "Unrecognized value, 'operation' should be one of the following: "
        )
        assert_pipeline_message = (
            "Unrecognized value, 'pipeline' should be one of the following: "
        )


class in_cluster:
    class models:
        assert_model_message = (
            "Unrecognized value, model should be one of the following: "
        )


class in_utils:
    class preprocessing:
        assert_imputation_categorical = "Unrecognized value, 'imputation_categorical' should be one of the following: "
        assert_scaler_message = (
            "Unrecognized value, 'scaler' should be one of the following: "
        )
        assert_imputation_numeric = (
            "Unrecognized value, 'imputation_numeric' should be one of the following: "
        )
        assert_one_hot_encoding = (
            "Unrecognized value, 'one_hot_encoding' should be one of the following: "
        )


######################
# Strings in pages #
######################


class in_page_classification_and_regression:
    title_message = ""
