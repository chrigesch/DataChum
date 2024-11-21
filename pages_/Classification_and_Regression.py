# Import moduls from local directories
from assets.colors import AVAILABLE_COLORS_SEQUENTIAL
from modules.classification_and_regression.cv_workflow import (
    AVAILABLE_SCORES_CLASSIFICATION,
    AVAILABLE_SCORES_REGRESSION,
)
from modules.classification_and_regression.evaluation import (
    corrected_repeated_t_test,
    _get_X_and_y_of_nested_cv,
    plot_pipeline,
    plot_learning_curve,
    plot_classification_calibration_curve,
    plot_classification_class_prediction,
    plot_classification_confusion_matrix,
    plot_classification_cumulative_gain,
    plot_classification_lift_curve,
    plot_classification_ks_statistic,
    plot_classification_precicion_recall,
    plot_classification_report,
    plot_classification_roc_curves,
    plot_classification_threshold,
    plot_regression_prediction_error,
    plot_regression_residuals,
    get_tuning_trials_dataframe,
    plot_tuning_optimization_history,
    plot_tuning_param_importances,
    plot_tuning_slice,
)
from modules.classification_and_regression.feature_selection import (
    AVAILABLE_FEATURE_SELECTION_METHODS,
)
from modules.classification_and_regression.interpretation import (
    compute_average_treatment_effect,
    partial_dependence_plot,
    plot_ate,
    compute_shap_values_agnostic,
    compute_shap_values_tree,
    plot_shap_beeswarm,
    plot_shap_feature_clustering,
    plot_shap_feature_importance,
    plot_shap_local,
    plot_shap_scatter,
)
from modules.classification_and_regression.main import (
    k_fold_cross_validation,
    nested_k_fold_cross_validation,
)
from modules.exploratory_data_analysis.univariate_and_bivariate import (
    plot_cat,
    plot_num,
)
from modules.utils.load_and_save_data import (
    convert_dataframe_to_csv,
    convert_dataframe_to_xlsx,
    read_csv,
    read_xlsx,
)
from modules.utils.preprocessing import (
    AVAILABLE_IMPUTATION_CATEGORICAL,
    AVAILABLE_IMPUTATION_NUMERICAL,
    AVAILABLE_SCALER,
    clean_strings_and_feature_names,
    _get_feature_names_after_preprocessing,
)

# Import the required libraries
import numpy as np
import pandas as pd
import pickle
import streamlit as st
import streamlit.components.v1 as components

# from streamlit_profiler import Profiler


def main():
    st.subheader("Classification and Regression")

    # Profile the app
    #    streamlit_profiler = Profiler()
    #    streamlit_profiler.start()

    # Copy data from session state
    if st.session_state.data is not None:
        data = st.session_state.data
        # Clean strings and feature names
        data = clean_strings_and_feature_names(data=data)
        # Get column names (also NUMERICAL and CATEGORICAL)
        cols_all = data.columns.to_list()
        cols_num = data.select_dtypes(include=["float", "int"]).columns.to_list()
        cols_cat = data.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.to_list()
        # Drop ID columns (or similar): Analyze whether all values of the categorical columns are unique
        # (count of unique values equals column's length)
        list_of_dropped_columns = []
        for column in cols_cat:
            if len(data[column]) == len(data[column].unique()):
                data = data.drop(column, axis=1)
                list_of_dropped_columns.append(column)
        if len(list_of_dropped_columns) > 0:
            #            st.markdown(f"""**{str(string_to_be_displayed)}**""")
            st.warning(
                "Following columns have been removed as all values of the column are unique: "
                + ", ".join(list_of_dropped_columns)
            )
            # Update cols_cat & cols_all
            cols_cat = data.select_dtypes(
                include=["object", "category", "bool"]
            ).columns.to_list()
            cols_all = data.columns.to_list()
        # Create three columns
        col_1, col_2, col_3 = st.columns(3)
        with col_1:
            # Create selectbox for the target variable and print if it is a regression or classification problem
            selectbox_target_variable = st.selectbox(
                label="**Select the target variable**", options=cols_all
            )
            if selectbox_target_variable in cols_cat:
                st.markdown("This is a **classification** task")
                operation = "classification"
            elif selectbox_target_variable in cols_num:
                st.markdown("This is a **regression** task")
                operation = "regression"
            st.markdown(
                "Please check if the data types of the features were infered correctly (Integers will be handled as numerical variables)"  # noqa: E501
            )
            if data[selectbox_target_variable].isna().sum() > 0:
                st.warning(
                    "This target variable contains "
                    + str(data[selectbox_target_variable].isna().sum())
                    + " missing values. These will be removed before starting the cross-validation"
                )
        with col_2:
            cols_cat_df = pd.DataFrame()
            cols_cat_df["Categorical Features"] = [
                value for value in cols_cat if value != selectbox_target_variable
            ]
            st.dataframe(cols_cat_df, use_container_width=True)
        with col_3:
            cols_num_df = pd.DataFrame()
            cols_num_df["Numerical Features"] = [
                value for value in cols_num if value != selectbox_target_variable
            ]
            st.dataframe(cols_num_df, use_container_width=True)
        st.subheader("**Predictive Modeling Setup**")
        # Create four tabs
        tab_s1, tab_s2, tab_s3, tab_s4 = st.tabs(
            [
                "**Procedure**",
                "**Preprocessing and Feature Selection**",
                "**Models**",
                "**Tuning**",
            ]
        )
        # Tab 1: Procedure
        with tab_s1:
            col_1, col_2, col_3 = st.columns([1.0, 1.5, 1.5])
            with col_1:
                selectbox_procedure = st.selectbox(
                    label="**Select the procedure**",
                    options=["k-fold crossvalidation", "nested crossvalidation"],
                )
                if selectbox_procedure == "k-fold crossvalidation":
                    selectbox_train_size = st.selectbox(
                        label="**Select train size**",
                        options=[
                            np.round(x, decimals=2) for x in np.arange(0.5, 0.95, 0.05)
                        ],
                        index=6,
                    )
                else:
                    selectbox_n_outer_cv_folds = st.selectbox(
                        label="**Select the number of OUTER folds**",
                        options=range(5, 11, 1),
                        index=5,
                    )
                    selectbox_n_outer_cv_reps = st.selectbox(
                        label="**Select the number of times OUTER cross-validator needs to be repeated**",
                        options=range(1, 11, 1),
                        index=4,
                    )
            with col_2:
                # Selectboxes for train size and/or number of folds and repetitions
                if selectbox_procedure == "k-fold crossvalidation":
                    selectbox_n_inner_cv_folds = st.selectbox(
                        label="**Select the number of k-folds**",
                        options=range(5, 11, 1),
                        index=5,
                    )
                    selectbox_n_inner_cv_reps = st.selectbox(
                        label="**Select the number of times cross-validator needs to be repeated**",
                        options=range(1, 11, 1),
                        index=4,
                    )
                else:
                    selectbox_n_inner_cv_folds = st.selectbox(
                        label="**Select the number of INNER folds**",
                        options=range(5, 11, 1),
                        index=5,
                    )
                    selectbox_n_inner_cv_reps = st.selectbox(
                        label="**Select the number of times INNER cross-validator needs to be repeated**",
                        options=range(1, 11, 1),
                        index=4,
                    )
            with col_3:
                # Selectboxes for evaluation metrics
                if operation == "classification":
                    selectbox_evaluation_score = st.selectbox(
                        label="**Select the evaluation score to be used in cross-validation**",
                        options=AVAILABLE_SCORES_CLASSIFICATION,
                        index=0,
                    )
                    selectbox_evaluation_average = st.selectbox(
                        label="**Select the average to compute AUC, Recall, Precision and F1**",
                        options=["micro", "macro", "weighted"],
                        index=1,
                    )
                else:
                    selectbox_evaluation_score = st.selectbox(
                        label="**Select the evaluation score to be used in cross-validation**",
                        options=AVAILABLE_SCORES_REGRESSION,
                        index=2,
                    )
                    # Placeholder to avoid error message
                    selectbox_evaluation_average = "macro"
        # Tab 2: Preprocessing and Feature Selection
        with tab_s2:
            col_1, col_2, col_3 = st.columns([1.5, 1.5, 1.0])
            with col_1:
                selectbox_imput_cat = st.selectbox(
                    label="**Select the imputation strategy to use for categorical variables**",
                    options=AVAILABLE_IMPUTATION_CATEGORICAL,
                    index=0,
                )
            with col_2:
                selectbox_imput_num = st.selectbox(
                    label="**Select the imputation strategy to use for numerical variables**",
                    options=AVAILABLE_IMPUTATION_NUMERICAL,
                    index=0,
                )
                selectbox_scaler = st.selectbox(
                    label="**Select the scaling strategy to use for numerical variables**",
                    options=AVAILABLE_SCALER,
                    index=4,
                )
            with col_3:
                selectbox_feature_selector = st.selectbox(
                    label="**Select a feature selection strategy**",
                    options=AVAILABLE_FEATURE_SELECTION_METHODS,
                    index=3,
                    help="**For more information of the Boruta method,** see Kumar, S. S., & Shaikh, T. (2017)."
                    " Empirical Evaluation of the Performance of Feature Selection Approaches on Random Forest."
                    " 2017 International Conference on Computer and Applications (ICCA), 227-231."
                    " https://doi.org/10.1109/COMAPP.2017.8079769 and Speiser, J. L., Miller, M. E., Tooze, J.,"
                    " & Ip, E. (2019). A comparison of random forest variable selection methods for"
                    " classification prediction modeling. Expert Systems with Applications, 134, 93-101."
                    " https://doi.org/10.1016/j.eswa.2019.05.028.  \n **For more information of the L1-based"
                    " linear support vector machine (L1-SVM) and LASSO,** see Sun, P., Wang, D., Mok, V. C.,"
                    " & Shi, L. (2019). Comparison of Feature Selection Methods and Machine Learning"
                    " Classifiers for Radiomics Analysis in Glioma Grading. IEEE Access, 7, 102010-102020."
                    " https://doi.org/10.1109/ACCESS.2019.2928975",
                )
        # Tab 3: Models
        with tab_s3:
            models_to_evaluate = []
            if operation == "classification":
                col_1, col_2, col_3, col_4 = st.columns(4)
                with col_1:
                    checkbox_1_1 = st.checkbox("LDA")
                    if checkbox_1_1:
                        models_to_evaluate.append("LDA")
                    checkbox_1_2 = st.checkbox("LogisticRegression")
                    if checkbox_1_2:
                        models_to_evaluate.append("LogisticRegression")
                    checkbox_1_3 = st.checkbox("Lasso")
                    if checkbox_1_3:
                        models_to_evaluate.append("Lasso")
                    checkbox_1_4 = st.checkbox("Ridge", value=True)
                    if checkbox_1_4:
                        models_to_evaluate.append("Ridge")
                    checkbox_1_5 = st.checkbox("ElasticNet")
                    if checkbox_1_5:
                        models_to_evaluate.append("ElasticNet")
                with col_2:
                    checkbox_5 = st.checkbox("GaussianNaiveBayes")
                    if checkbox_5:
                        models_to_evaluate.append("GaussianNaiveBayes")
                    checkbox_6 = st.checkbox("KNN")
                    if checkbox_6:
                        models_to_evaluate.append("KNN")
                    checkbox_7 = st.checkbox("SVM_L")
                    if checkbox_7:
                        models_to_evaluate.append("SVM_L")
                    checkbox_8 = st.checkbox("SVM_R")
                    if checkbox_8:
                        models_to_evaluate.append("SVM_R")
                with col_3:
                    checkbox_9 = st.checkbox("DecisionTree")
                    if checkbox_9:
                        models_to_evaluate.append("DecisionTree")
                    checkbox_10 = st.checkbox("RandomForest")
                    if checkbox_10:
                        models_to_evaluate.append("RandomForest")
                    checkbox_11 = st.checkbox("ExtraTrees")
                    if checkbox_11:
                        models_to_evaluate.append("ExtraTrees")
                with col_4:
                    checkbox_12 = st.checkbox("CatBoost")
                    if checkbox_12:
                        models_to_evaluate.append("CatBoost")
                    checkbox_13 = st.checkbox("LightGBM", value=True)
                    if checkbox_13:
                        models_to_evaluate.append("LightGBM")
                    checkbox_14 = st.checkbox("XGB")
                    if checkbox_14:
                        models_to_evaluate.append("XGB")
                    checkbox_15 = st.checkbox("TensorFlow")
                    if checkbox_15:
                        models_to_evaluate.append("TensorFlow")
            else:
                col_1, col_2, col_3, col_4 = st.columns(4)
                with col_1:
                    checkbox_1 = st.checkbox("Bayesian_Ridge")
                    if checkbox_1:
                        models_to_evaluate.append("Bayesian_Ridge")
                    checkbox_2 = st.checkbox("Linear_Regression", value=True)
                    if checkbox_2:
                        models_to_evaluate.append("Linear_Regression")
                    checkbox_3 = st.checkbox("KNN")
                    if checkbox_3:
                        models_to_evaluate.append("KNN")
                with col_2:
                    checkbox_4 = st.checkbox("SVM_L")
                    if checkbox_4:
                        models_to_evaluate.append("SVM_L")
                    checkbox_5 = st.checkbox("SVM_R")
                    if checkbox_5:
                        models_to_evaluate.append("SVM_R")
                with col_3:
                    checkbox_6 = st.checkbox("DecisionTree")
                    if checkbox_6:
                        models_to_evaluate.append("DecisionTree")
                    checkbox_7 = st.checkbox("RandomForest")
                    if checkbox_7:
                        models_to_evaluate.append("RandomForest")
                    checkbox_8 = st.checkbox("ExtraTrees")
                    if checkbox_8:
                        models_to_evaluate.append("ExtraTrees")
                with col_4:
                    checkbox_9 = st.checkbox("CatBoost")
                    if checkbox_9:
                        models_to_evaluate.append("CatBoost")
                    checkbox_10 = st.checkbox("LightGBM", value=True)
                    if checkbox_10:
                        models_to_evaluate.append("LightGBM")
                    checkbox_11 = st.checkbox("XGB")
                    if checkbox_11:
                        models_to_evaluate.append("XGB")
                    checkbox_12 = st.checkbox("TensorFlow")
                    if checkbox_12:
                        models_to_evaluate.append("TensorFlow")
        # Tab 4: Tuning
        with tab_s4:
            # Assign default values:
            selectbox_tune_imp_categorical = False
            selectbox_tune_imp_numerical = False
            selectbox_tune_scaler = False
            # Create two columns
            col_1, col_2 = st.columns(2)
            with col_1:
                selectbox_n_tuning_trials = st.selectbox(
                    label="**Select the number of tuning trials**",
                    options=range(0, 501, 1),
                    index=0,
                )
                selectbox_cv_with_pipeline = st.selectbox(
                    label="**Perform preprocessing (and feature selection) within each k-fold (and tuning trial)**",
                    options=[True, False],
                    index=1,
                )
            if selectbox_cv_with_pipeline is True:
                with col_2:
                    selectbox_tune_imp_categorical = st.selectbox(
                        label="**Include the categorical imputer as a hyperparameter during the tuning process**",
                        options=[True, False],
                        index=1,
                    )
                    selectbox_tune_imp_numerical = st.selectbox(
                        label="**Include the numerical imputer as a hyperparameter during the tuning process**",
                        options=[True, False],
                        index=1,
                    )
                    selectbox_tune_scaler = st.selectbox(
                        label="**Include the scaler as a hyperparameter during the tuning process**",
                        options=[True, False],
                        index=1,
                    )

        # Instantiate placeholders | Session state variables
        if "cv_instance" not in st.session_state:
            st.session_state.cv_instance = None
        # Create a button to run the cross-validation
        button_run_cv = st.button(
            label="**Run cross-validation**", type="primary", use_container_width=True
        )
        if button_run_cv:
            with st.spinner("Training the selected models..."):
                if selectbox_procedure == "k-fold crossvalidation":
                    st.session_state.cv_instance = k_fold_cross_validation(
                        operation=operation,
                        data=data,
                        target_variable=selectbox_target_variable,
                        train_size=selectbox_train_size,
                        imputation_numerical=selectbox_imput_num,
                        imputation_categorical=selectbox_imput_cat,
                        scaler=selectbox_scaler,
                        one_hot_encoding=True,
                        feature_selection=selectbox_feature_selector,
                        models_to_be_evaluated=models_to_evaluate,
                        cv_with_pipeline=selectbox_cv_with_pipeline,
                        inner_cv_folds=selectbox_n_inner_cv_folds,
                        inner_cv_rep=selectbox_n_inner_cv_reps,
                        tuning_trials=selectbox_n_tuning_trials,
                        evaluation_score=selectbox_evaluation_score,
                        average=selectbox_evaluation_average,
                        tune_imp_numerical=selectbox_tune_imp_numerical,
                        tune_scaler=selectbox_tune_scaler,
                        tune_imp_categorical=selectbox_tune_imp_categorical,
                    )
                else:
                    st.session_state.cv_instance = nested_k_fold_cross_validation(
                        operation=operation,
                        data=data,
                        target_variable=selectbox_target_variable,
                        imputation_numerical=selectbox_imput_num,
                        imputation_categorical=selectbox_imput_cat,
                        scaler=selectbox_scaler,
                        one_hot_encoding=True,
                        feature_selection=selectbox_feature_selector,
                        models_to_be_evaluated=models_to_evaluate,
                        cv_with_pipeline=selectbox_cv_with_pipeline,
                        outer_cv_folds=selectbox_n_outer_cv_folds,
                        outer_cv_rep=selectbox_n_outer_cv_reps,
                        inner_cv_folds=selectbox_n_inner_cv_folds,
                        inner_cv_rep=selectbox_n_inner_cv_reps,
                        tuning_trials=selectbox_n_tuning_trials,
                        evaluation_score=selectbox_evaluation_score,
                        average=selectbox_evaluation_average,
                        tune_imp_numerical=selectbox_tune_imp_numerical,
                        tune_scaler=selectbox_tune_scaler,
                        tune_imp_categorical=selectbox_tune_imp_categorical,
                    )
            st.success("Done!")
            # Rerun app (tab hack: prevent jumping back to first tab)
            # https://github.com/streamlit/streamlit/issues/4996#issuecomment-1261464494
            st.experimental_rerun()

        if st.session_state.cv_instance is not None:
            st.subheader("**Model Evaluation**")
            # Create three tabs
            tab_e1_1, tab_e1_2, tab_e1_3, tab_e1_4, tab_e1_5 = st.tabs(
                [
                    "**Cross-validation Scores**",
                    "**Test Scores**",
                    "**Plot Scores**",
                    "**Evaluation**",
                    "**Interpretation**",
                ]
            )
            # Tab 1: Cross-validation Scores
            with tab_e1_1:
                # Create DataFrame with cross-validation scores and number of samples in X_train
                scores_cv_df = st.session_state.cv_instance.all_results_cv
                if st.session_state.cv_instance.procedure == "k_fold":
                    n_X_train = len(st.session_state.cv_instance.X_train)
                else:
                    n_X_train = int(
                        len(st.session_state.cv_instance.data)
                        * (1 - 1 / st.session_state.cv_instance.outer_cv_folds)
                    )
                # Create two columns for the download buttons
                col_cv_score_1_1, col_cv_score_1_2 = st.columns([1, 2])
                with col_cv_score_1_1:
                    st.download_button(
                        label="Download cross-validation scores as CSV",
                        data=convert_dataframe_to_csv(scores_cv_df),
                        file_name="scores_cv.csv",
                        mime="text/csv'",
                    )
                with col_cv_score_1_2:
                    st.download_button(
                        label="Download cross-validation scores as XLSX",
                        data=convert_dataframe_to_xlsx(scores_cv_df),
                        file_name="scores_cv.xlsx",
                        mime="application/vnd.ms-excel",
                    )
                # Create tabs to display results and to compute t-tests
                col_cv_score_2_1, col_cv_score_2_2 = st.columns([2.0, 1.0])
                # Display cross-validation results
                with col_cv_score_2_1:
                    st.markdown("**Grouped by model - means**")
                    st.dataframe(
                        scores_cv_df.groupby(by="model").mean().style.format("{:.3f}"),
                        height=(
                            (len(scores_cv_df.groupby(by="model").mean()) + 1) * 35 + 3
                        ),
                        use_container_width=False,
                    )
                    st.markdown("**Complete**")
                    st.dataframe(
                        scores_cv_df.set_index("model").style.format("{:.3f}"),
                        use_container_width=False,
                    )
                # Compute and display t-test results
                with col_cv_score_2_2:
                    st.markdown(
                        "**Corrected Repeated t-test**",
                        help="**It is recommended to run 10-times 10-fold cross-validation.**"
                        " For more information, see:  \nNadeau, C., & Bengio, Y. (2003). Inference for the"
                        " generalization error. Machine Learning, 52(3), 239-281."
                        " https://doi.org/10.1023/A:1024068626366.  \nBouckaert, R. R., & Frank, E. (2004)."
                        " Evaluating the Replicability of Significance Tests for Comparing Learning Algorithms."
                        " In H. Dai, R. Srikant, & C. Zhang (Eds.), Advances in Knowledge Discovery and Data Mining."
                        " PAKDD 2004. Lecture Notes in Computer Science, vol 3056 (pp. 3-12). Springer."
                        " https://doi.org/10.1007/978-3-540-24775-3_3",
                    )
                    models_to_be_compared = scores_cv_df["model"].unique().tolist()
                    if len(models_to_be_compared) < 2:
                        st.markdown(
                            "Only available if two ore more models were evaluated"
                        )
                    else:
                        selectbox_t_test_model_1 = st.selectbox(
                            label="**Select model 1 to be compared**",
                            options=models_to_be_compared,
                            index=0,
                            key="t_test_cv_model_1",
                        )
                        selectbox_t_test_model_2 = st.selectbox(
                            label="**Select model 2 to be compared**",
                            options=[
                                value
                                for value in models_to_be_compared
                                if value != selectbox_t_test_model_1
                            ],
                            index=0,
                            key="t_test_cv_model_2",
                        )
                        # Extract scores to compute corrected_repeated_t_test
                        evaluation_metric = (
                            st.session_state.cv_instance.evaluation_score
                        )
                        scores_model_1 = scores_cv_df[
                            scores_cv_df["model"] == selectbox_t_test_model_1
                        ][evaluation_metric]
                        scores_model_2 = scores_cv_df[
                            scores_cv_df["model"] == selectbox_t_test_model_2
                        ][evaluation_metric]

                        result_t_test = corrected_repeated_t_test(
                            scores_model_1=scores_model_1,
                            scores_model_2=scores_model_2,
                            n_folds=st.session_state.cv_instance.inner_cv_folds,
                            n=n_X_train,
                        )
                        # Change model names
                        result_t_test.result_descriptives["model"] = [
                            selectbox_t_test_model_1,
                            selectbox_t_test_model_2,
                        ]
                        st.markdown(f"""**Descriptives - {str(evaluation_metric)}**""")
                        st.dataframe(
                            result_t_test.result_descriptives.set_index(
                                "model"
                            ).style.format("{:.3f}"),
                            use_container_width=False,
                        )
                        st.markdown("**Statistics**")
                        st.dataframe(
                            result_t_test.result_statistics.set_index(
                                "t_statistic"
                            ).style.format("{:.3f}"),
                            use_container_width=False,
                        )
            # Tab 2: Test Scores
            with tab_e1_2:
                # Create DataFrame with test scores
                scores_test_df = st.session_state.cv_instance.all_results_test
                # Create two columns for the download buttons
                col_test_score_1_1, col_test_score_1_2 = st.columns([1, 2])
                with col_test_score_1_1:
                    st.download_button(
                        label="Download test scores as CSV",
                        data=convert_dataframe_to_csv(scores_test_df),
                        file_name="scores_test.csv",
                        mime="text/csv'",
                    )
                with col_test_score_1_2:
                    st.download_button(
                        label="Download test scores as XLSX",
                        data=convert_dataframe_to_xlsx(scores_test_df),
                        file_name="scores_test.xlsx",
                        mime="application/vnd.ms-excel",
                    )
                # Create tabs to display results and to compute t-tests
                col_test_score_2_1, col_test_score_2_2 = st.columns([2.0, 1.0])
                # Display cross-validation results
                with col_test_score_2_1:
                    st.markdown("**Grouped by model - means**")
                    st.dataframe(
                        scores_test_df.groupby(by="model")
                        .mean()
                        .style.format("{:.3f}"),
                        height=(
                            (len(scores_test_df.groupby(by="model").mean()) + 1) * 35
                            + 3
                        ),
                        use_container_width=False,
                    )
                    st.markdown("**Complete**")
                    st.dataframe(
                        scores_test_df.set_index("model").style.format("{:.3f}"),
                        use_container_width=False,
                    )
                # Compute and display t-test results
                if st.session_state.cv_instance.procedure == "nested":
                    with col_test_score_2_2:
                        st.markdown(
                            "**Corrected Repeated t-test**",
                            help="**It is recommended to run 10-times 10-fold cross-validation.**"
                            " For more information, see:  \nNadeau, C., & Bengio, Y. (2003). Inference for the"
                            " generalization error. Machine Learning, 52(3), 239-281."
                            " https://doi.org/10.1023/A:1024068626366.  \nBouckaert, R. R., & Frank, E. (2004)."
                            " Evaluating the Replicability of Significance Tests for Comparing Learning Algorithms."
                            " In H. Dai, R. Srikant, & C. Zhang (Eds.), Advances in Knowledge Discovery and Data Mining."  # noqa E:501
                            " PAKDD 2004. Lecture Notes in Computer Science, vol 3056 (pp. 3-12). Springer."
                            " https://doi.org/10.1007/978-3-540-24775-3_3",
                        )
                        models_to_be_compared = (
                            scores_test_df["model"].unique().tolist()
                        )
                        if len(models_to_be_compared) < 2:
                            st.markdown(
                                "Only available if two ore more models were evaluated"
                            )
                        else:
                            selectbox_t_test_model_1 = st.selectbox(
                                label="**Select model 1 to be compared**",
                                options=models_to_be_compared,
                                index=0,
                                key="t_test_test_model_1",
                            )
                            selectbox_t_test_model_2 = st.selectbox(
                                label="**Select model 2 to be compared**",
                                options=[
                                    value
                                    for value in models_to_be_compared
                                    if value != selectbox_t_test_model_1
                                ],
                                index=0,
                                key="t_test_test_model_2",
                            )
                            # Extract scores to compute corrected_repeated_t_test
                            evaluation_metric = (
                                st.session_state.cv_instance.evaluation_score
                            )
                            scores_model_1 = scores_test_df[
                                scores_cv_df["model"] == selectbox_t_test_model_1
                            ][evaluation_metric]
                            scores_model_2 = scores_test_df[
                                scores_cv_df["model"] == selectbox_t_test_model_2
                            ][evaluation_metric]

                            result_t_test = corrected_repeated_t_test(
                                scores_model_1=scores_model_1,
                                scores_model_2=scores_model_2,
                                n_folds=st.session_state.cv_instance.outer_cv_folds,
                                n=len(st.session_state.cv_instance.data),
                            )
                            # Change model names
                            result_t_test.result_descriptives["model"] = [
                                selectbox_t_test_model_1,
                                selectbox_t_test_model_2,
                            ]
                            st.markdown(
                                f"""**Descriptives - {str(evaluation_metric)}**"""
                            )
                            # Display results
                            st.dataframe(
                                result_t_test.result_descriptives.set_index(
                                    "model"
                                ).style.format("{:.3f}"),
                                use_container_width=False,
                            )
                            st.markdown("**Statistics**")
                            st.dataframe(
                                result_t_test.result_statistics.set_index(
                                    "t_statistic"
                                ).style.format("{:.3f}"),
                                use_container_width=False,
                            )

            # Tab 3: Plot Scores
            with tab_e1_3:
                # Create two columns: to select a model and to display it)
                col_ps_1_1, col_ps_1_2 = st.columns([1, 3])
                with col_ps_1_1:
                    selectbox_scores = st.selectbox(
                        label="**Select which scores to plot**",
                        options=["Cross-validation", "Test"],
                        index=0,
                        key="col_ps_1_1_selectbox_1",
                    )
                    selectbox_evaluation_metric = st.selectbox(
                        label="**Select the evaluation metric to be plotted**",
                        options=[
                            value
                            for value in scores_cv_df.columns.to_list()
                            if value not in ["model", "time"]
                        ],
                        index=0,
                        key="col_ps_1_1_selectbox_2",
                    )
                    selectbox_color = st.selectbox(
                        label="**Select a color scale**",
                        options=AVAILABLE_COLORS_SEQUENTIAL,
                        index=0,
                        key="col_ps_1_1_selectbox_3",
                    )
                with col_ps_1_2:
                    if selectbox_scores == "Cross-validation":
                        fig_variable = plot_num(
                            data=scores_cv_df,
                            var_num=selectbox_evaluation_metric,
                            var_cat="model",
                            plot_type="Box-Plot",
                            color=selectbox_color,
                            template="plotly_white",
                        )
                        st.plotly_chart(
                            fig_variable,
                            theme="streamlit",
                            use_container_width=True,
                        )
                    elif (selectbox_scores == "Test") & (
                        st.session_state.cv_instance.procedure == "nested"
                    ):
                        fig_variable = plot_num(
                            data=scores_test_df,
                            var_num=selectbox_evaluation_metric,
                            var_cat="model",
                            plot_type="Box-Plot",
                            color=selectbox_color,
                            template="plotly_white",
                        )
                        st.plotly_chart(
                            fig_variable,
                            theme="streamlit",
                            use_container_width=True,
                        )
                    else:
                        fig_variable = plot_cat(
                            data=scores_test_df,
                            var_cat="model",
                            var_num=selectbox_evaluation_metric,
                            plot_type="Bar",
                            color=selectbox_color,
                            template="plotly_white",
                        )
                        st.plotly_chart(
                            fig_variable, theme="streamlit", use_container_width=True
                        )

            # Tab 4: Evaluation
            with tab_e1_4:
                # Create two columns: to select a model and to display it)
                col_ep_1_1, col_ep_1_2 = st.columns([1, 3])
                # Column 1: Display the available models and select the model to be plotted
                with col_ep_1_1:
                    selectbox_model_to_be_plotted = st.selectbox(
                        label="**Select the index number of the model to be plotted or downloaded**",
                        options=range(len(scores_test_df["model"])),
                        index=0,
                    )
                    if st.session_state.cv_instance.procedure == "k_fold":
                        st.dataframe(
                            scores_test_df["model"].reset_index(drop=True),
                            use_container_width=True,
                        )
                    else:
                        st.dataframe(
                            scores_test_df[["model", "rep", "fold"]],
                            use_container_width=True,
                        )
                    # Extract: models (a)
                    models_trained_list = (
                        st.session_state.cv_instance.all_results_models
                    )
                    # Extract: all data (b)
                    if st.session_state.cv_instance.procedure == "k_fold":
                        X_train = st.session_state.cv_instance.X_train
                        X_test = st.session_state.cv_instance.X_test
                        y_train = st.session_state.cv_instance.y_train
                        y_test = st.session_state.cv_instance.y_test
                    else:
                        X_and_y_instance = _get_X_and_y_of_nested_cv(
                            X=st.session_state.cv_instance.X,
                            y=st.session_state.cv_instance.y,
                            outer_cv_object=st.session_state.cv_instance.outer_cv_object,
                            outer_cv_folds=st.session_state.cv_instance.outer_cv_folds,
                            needed_cv_rep=scores_test_df["rep"][
                                selectbox_model_to_be_plotted
                            ],
                            needed_cv_fold=scores_test_df["fold"][
                                selectbox_model_to_be_plotted
                            ],
                        )
                        X_train = X_and_y_instance.X_train
                        X_test = X_and_y_instance.X_test
                        y_train = X_and_y_instance.y_train
                        y_test = X_and_y_instance.y_test
                    # Extract: label_encoder (c) (or placeholder, if 'regression')
                    if st.session_state.cv_instance.operation == "classification":
                        label_encoder = st.session_state.cv_instance.label_encoder
                        # Determine if it is a binary classification task
                        if (
                            len(
                                st.session_state.cv_instance.data[
                                    st.session_state.cv_instance.target_variable
                                ]
                                .dropna()
                                .unique()
                            )
                            == 2
                        ):
                            binary_classification = True
                        else:
                            binary_classification = False
                    else:
                        label_encoder = None
                    # Extract: tuning information (d)
                    if st.session_state.cv_instance.tuning_trials > 0:
                        tuning_results = st.session_state.cv_instance.all_results_tuning
                    # Download bottons for models and label encoder
                    st.download_button(
                        label="Download model with its pipeline",
                        data=pickle.dumps(
                            models_trained_list[selectbox_model_to_be_plotted]
                        ),
                        file_name="pipeline_"
                        + str(
                            scores_test_df["model"].iloc[selectbox_model_to_be_plotted]
                        )
                        + ".pkl",
                        key="download_model_eval",
                        use_container_width=True,
                    )
                    if st.session_state.cv_instance.operation == "classification":
                        st.download_button(
                            label="Download label encoder",
                            data=pickle.dumps(label_encoder),
                            file_name="label_encoder_"
                            + str(
                                scores_test_df["model"].iloc[
                                    selectbox_model_to_be_plotted
                                ]
                            )
                            + ".pkl",
                            key="download_label_encoder_eval",
                            use_container_width=True,
                        )
                # Column 2: Display the available plots and plot them
                with col_ep_1_2:
                    # Create markdown and five columns for Generic Plots
                    st.markdown("**Generic Plots**")
                    (
                        col_ep_2_1_1,
                        col_ep_2_1_2,
                        col_ep_2_1_3,
                        col_ep_2_1_4,
                        col_ep_2_1_5,
                    ) = st.columns(5)
                    with col_ep_2_1_1:
                        button_pipeline = st.button(
                            label="**Pipeline**",
                            type="secondary",
                            use_container_width=True,
                        )
                    with col_ep_2_1_2:
                        button_learning_curve = st.button(
                            label="**Learning Curve**",
                            type="secondary",
                            use_container_width=True,
                        )
                    # Create markdown and five columns for Classification Plots
                    if st.session_state.cv_instance.operation == "classification":
                        st.markdown("**Classification Plots**")
                        (
                            col_ep_2_2_1,
                            col_ep_2_2_2,
                            col_ep_2_2_3,
                            col_ep_2_2_4,
                            col_ep_2_2_5,
                        ) = st.columns(5)
                        with col_ep_2_2_1:
                            button_class_prediction = st.button(
                                label="**Class Prediction**",
                                type="secondary",
                                use_container_width=True,
                            )
                        with col_ep_2_2_2:
                            button_confusion_matrix = st.button(
                                label="**Confusion Matrix**",
                                type="secondary",
                                use_container_width=True,
                            )
                        with col_ep_2_2_3:
                            button_precicion_recall_curve = st.button(
                                label="**Precision Recall Curve**",
                                type="secondary",
                                use_container_width=True,
                            )
                        with col_ep_2_2_4:
                            button_classification_report = st.button(
                                label="**Classification Report**",
                                type="secondary",
                                use_container_width=True,
                            )
                        with col_ep_2_2_5:
                            button_roc_curves = st.button(
                                label="**ROC Curves**",
                                type="secondary",
                                use_container_width=True,
                            )

                        # Create five columns for Classification Plots
                        # that are only available for Binary target variables
                        if binary_classification is True:
                            (
                                col_ep_2_3_1,
                                col_ep_2_3_2,
                                col_ep_2_3_3,
                                col_ep_2_3_4,
                                col_ep_2_3_5,
                            ) = st.columns(5)
                            with col_ep_2_3_1:
                                button_calibration_curve = st.button(
                                    label="**Calibration Curve**",
                                    type="secondary",
                                    use_container_width=True,
                                )
                            with col_ep_2_3_2:
                                button_cumulative_gain = st.button(
                                    label="**Cumulative Gain**",
                                    type="secondary",
                                    use_container_width=True,
                                )
                            with col_ep_2_3_3:
                                button_lift_curve = st.button(
                                    label="**Lift Curve**",
                                    type="secondary",
                                    use_container_width=True,
                                )
                            with col_ep_2_3_4:
                                button_ks_statistic = st.button(
                                    label="**KS Statistics**",
                                    type="secondary",
                                    use_container_width=True,
                                )
                            with col_ep_2_3_5:
                                button_threshold = st.button(
                                    label="**Threshold**",
                                    type="secondary",
                                    use_container_width=True,
                                )
                    # Create markdown and five columns for Regression Plots
                    else:
                        st.markdown("**Regression Plots**")
                        (
                            col_ep_2_4_1,
                            col_ep_2_4_2,
                            col_ep_2_4_3,
                            col_ep_2_4_4,
                            col_ep_2_4_5,
                        ) = st.columns(5)
                        with col_ep_2_4_1:
                            button_prediction_error = st.button(
                                label="**Prediction Error**",
                                type="secondary",
                                use_container_width=True,
                            )
                        with col_ep_2_4_2:
                            button_regression_residuals = st.button(
                                label="**Regression Residuals**",
                                type="secondary",
                                use_container_width=True,
                            )
                    # Create markdown and five columns for Tuning Plots
                    if (
                        scores_test_df["model"].iloc[selectbox_model_to_be_plotted][-6:]
                        == "_tuned"
                    ):
                        st.markdown("**Tuning Plots**")
                        (
                            col_ep_2_5_1,
                            col_ep_2_5_2,
                            col_ep_2_5_3,
                            col_ep_2_5_4,
                            col_ep_2_5_5,
                        ) = st.columns(5)
                        with col_ep_2_5_1:
                            button_trials_dataframe = st.button(
                                label="**Trials DataFrame**",
                                type="secondary",
                                use_container_width=True,
                            )
                        with col_ep_2_5_2:
                            button_optimization_history = st.button(
                                label="**Optimization History**",
                                type="secondary",
                                use_container_width=True,
                            )
                        with col_ep_2_5_3:
                            button_param_importances = st.button(
                                label="**Parameter Importances**",
                                type="secondary",
                                use_container_width=True,
                            )
                        with col_ep_2_5_4:
                            button_tuning_slices = st.button(
                                label="**Tuning Slices**",
                                type="secondary",
                                use_container_width=True,
                            )
                    # Plot - Generic Plots
                    if button_pipeline:
                        fig_html = plot_pipeline(
                            models_trained_list[selectbox_model_to_be_plotted]
                        )
                        components.html(fig_html, height=400, width=600)
                    if button_learning_curve:
                        fig_html = plot_learning_curve(
                            pipeline=models_trained_list[selectbox_model_to_be_plotted],
                            data=data,
                            target_variable=st.session_state.cv_instance.target_variable,
                            operation=st.session_state.cv_instance.operation,
                            evaluation_score=st.session_state.cv_instance.evaluation_score,
                            average=st.session_state.cv_instance.average,
                            label_encoder=label_encoder,
                            cv_folds=10,
                            cv_rep=1,
                        )
                        components.html(fig_html, height=600)
                    # Plot - Classification Plots
                    if st.session_state.cv_instance.operation == "classification":
                        if button_class_prediction:
                            fig_variable = plot_classification_class_prediction(
                                pipeline=models_trained_list[
                                    selectbox_model_to_be_plotted
                                ],
                                label_encoder=label_encoder,
                                X_test=X_test,
                                y_test=y_test,
                            )
                            st.plotly_chart(
                                fig_variable,
                                theme="streamlit",
                                use_container_width=True,
                            )
                        if button_confusion_matrix:
                            fig_variable = plot_classification_confusion_matrix(
                                pipeline=models_trained_list[
                                    selectbox_model_to_be_plotted
                                ],
                                label_encoder=label_encoder,
                                X_test=X_test,
                                y_test=y_test,
                            )
                            st.plotly_chart(
                                fig_variable,
                                theme="streamlit",
                                use_container_width=True,
                            )
                        if button_precicion_recall_curve:
                            fig_html = plot_classification_precicion_recall(
                                pipeline=models_trained_list[
                                    selectbox_model_to_be_plotted
                                ],
                                label_encoder=label_encoder,
                                X_train=X_train,
                                X_test=X_test,
                                y_train=y_train,
                                y_test=y_test,
                            )
                            components.html(fig_html, height=600)
                        if button_classification_report:
                            fig_html = plot_classification_report(
                                pipeline=models_trained_list[
                                    selectbox_model_to_be_plotted
                                ],
                                label_encoder=label_encoder,
                                X_train=X_train,
                                X_test=X_test,
                                y_train=y_train,
                                y_test=y_test,
                            )
                            components.html(fig_html, height=600)
                        if button_roc_curves:
                            fig_html = plot_classification_roc_curves(
                                pipeline=models_trained_list[
                                    selectbox_model_to_be_plotted
                                ],
                                label_encoder=label_encoder,
                                X_train=X_train,
                                X_test=X_test,
                                y_train=y_train,
                                y_test=y_test,
                            )
                            components.html(fig_html, height=600)

                        # Classification Plots (if binary)
                        if binary_classification is True:
                            if button_calibration_curve:
                                fig_html = plot_classification_calibration_curve(
                                    pipeline=models_trained_list[
                                        selectbox_model_to_be_plotted
                                    ],
                                    X_test=X_test,
                                    y_test=y_test,
                                )
                                components.html(fig_html, height=600)
                            if button_cumulative_gain:
                                fig_html = plot_classification_cumulative_gain(
                                    pipeline=models_trained_list[
                                        selectbox_model_to_be_plotted
                                    ],
                                    label_encoder=label_encoder,
                                    X_test=X_test,
                                    y_test=y_test,
                                )
                                components.html(fig_html, height=600)
                            if button_lift_curve:
                                fig_html = plot_classification_lift_curve(
                                    pipeline=models_trained_list[
                                        selectbox_model_to_be_plotted
                                    ],
                                    label_encoder=label_encoder,
                                    X_test=X_test,
                                    y_test=y_test,
                                )
                                components.html(fig_html, height=600)
                            if button_ks_statistic:
                                fig_html = plot_classification_ks_statistic(
                                    pipeline=models_trained_list[
                                        selectbox_model_to_be_plotted
                                    ],
                                    label_encoder=label_encoder,
                                    X_test=X_test,
                                    y_test=y_test,
                                )
                                components.html(fig_html, height=600)
                            if button_threshold:
                                fig_html = plot_classification_threshold(
                                    pipeline=models_trained_list[
                                        selectbox_model_to_be_plotted
                                    ],
                                    label_encoder=label_encoder,
                                    X_train=X_train,
                                    X_test=X_test,
                                    y_train=y_train,
                                    y_test=y_test,
                                )
                                components.html(fig_html, height=600)
                    # Plot - Regression Plots
                    if st.session_state.cv_instance.operation == "regression":
                        if button_prediction_error:
                            fig_html = plot_regression_prediction_error(
                                pipeline=models_trained_list[
                                    selectbox_model_to_be_plotted
                                ],
                                X_train=X_train,
                                X_test=X_test,
                                y_train=y_train,
                                y_test=y_test,
                            )
                            components.html(fig_html, height=600)

                        if button_regression_residuals:
                            fig_html = plot_regression_residuals(
                                pipeline=models_trained_list[
                                    selectbox_model_to_be_plotted
                                ],
                                X_train=X_train,
                                X_test=X_test,
                                y_train=y_train,
                                y_test=y_test,
                            )
                            components.html(fig_html, height=600)
                    # Plot - Tuning Plots
                    if (
                        scores_test_df["model"].iloc[selectbox_model_to_be_plotted][-6:]
                        == "_tuned"
                    ):
                        if button_trials_dataframe:
                            tuning_trials_df = get_tuning_trials_dataframe(
                                tuning_results[selectbox_model_to_be_plotted]
                            )
                            # Convert 'timedelta' to 'int' (streamlit bug with st.dataframe()):
                            # https://github.com/streamlit/streamlit/issues/4489
                            for column in tuning_trials_df.select_dtypes(
                                include=["timedelta"]
                            ).columns.to_list():
                                tuning_trials_df[column] = tuning_trials_df[
                                    column
                                ].astype(
                                    "int"
                                )  # 'int' | 'complex128'
                                tuning_trials_df[column] = (
                                    tuning_trials_df[column] / 10000000000
                                )
                            tuning_trials_df = tuning_trials_df.drop(
                                ["datetime_start", "datetime_complete", "state"], axis=1
                            )
                            # Create dictionary to format floats
                            format_dict = {}
                            for column in tuning_trials_df.select_dtypes(
                                include=["float"]
                            ).columns.to_list():
                                format_dict[column] = "{:.3f}"
                            st.dataframe(tuning_trials_df.style.format(format_dict))
                        if button_optimization_history:
                            fig_variable = plot_tuning_optimization_history(
                                tuning_results[selectbox_model_to_be_plotted]
                            )
                            st.plotly_chart(
                                fig_variable,
                                theme="streamlit",
                                use_container_width=True,
                            )
                        if button_param_importances:
                            fig_variable = plot_tuning_param_importances(
                                tuning_results[selectbox_model_to_be_plotted]
                            )
                            st.plotly_chart(
                                fig_variable,
                                theme="streamlit",
                                use_container_width=True,
                            )
                        if button_tuning_slices:
                            fig_variable = plot_tuning_slice(
                                tuning_results[selectbox_model_to_be_plotted]
                            )
                            st.plotly_chart(
                                fig_variable,
                                theme="streamlit",
                                use_container_width=True,
                            )
            # Tab 5: Interpretation
            with tab_e1_5:
                # Create two columns: to select a model and to display it)
                col_ip_1_1, col_ip_1_2 = st.columns([1, 3])
                # Column 1: Display the available models and select the model to be plotted
                with col_ip_1_1:
                    selectbox_model_to_be_plotted_int = st.selectbox(
                        label="**Select the index number of the model to be plotted or downloaded**",
                        options=range(len(scores_test_df["model"])),
                        index=0,
                        key="model_interpretation_plots",
                    )
                    if st.session_state.cv_instance.procedure == "k_fold":
                        st.dataframe(
                            scores_test_df["model"].reset_index(drop=True),
                            use_container_width=True,
                        )
                    else:
                        st.dataframe(
                            scores_test_df[["model", "rep", "fold"]],
                            use_container_width=True,
                        )
                    # Extract: models (a)
                    models_trained_list = (
                        st.session_state.cv_instance.all_results_models
                    )
                    # Extract: all data (b)
                    if st.session_state.cv_instance.procedure == "k_fold":
                        X_train = st.session_state.cv_instance.X_train
                        X_test = st.session_state.cv_instance.X_test
                        y_train = st.session_state.cv_instance.y_train
                        y_test = st.session_state.cv_instance.y_test
                    else:
                        X_and_y_instance = _get_X_and_y_of_nested_cv(
                            X=st.session_state.cv_instance.X,
                            y=st.session_state.cv_instance.y,
                            outer_cv_object=st.session_state.cv_instance.outer_cv_object,
                            outer_cv_folds=st.session_state.cv_instance.outer_cv_folds,
                            needed_cv_rep=scores_test_df["rep"][
                                selectbox_model_to_be_plotted_int
                            ],
                            needed_cv_fold=scores_test_df["fold"][
                                selectbox_model_to_be_plotted_int
                            ],
                        )
                        X_train = X_and_y_instance.X_train
                        X_test = X_and_y_instance.X_test
                        y_train = X_and_y_instance.y_train
                        y_test = X_and_y_instance.y_test
                    # Extract: label_encoder (c) (or placeholder, if 'regression')
                    if st.session_state.cv_instance.operation == "classification":
                        label_encoder = st.session_state.cv_instance.label_encoder
                        # Determine if it is a binary classification task
                        if (
                            len(
                                st.session_state.cv_instance.data[
                                    st.session_state.cv_instance.target_variable
                                ]
                                .dropna()
                                .unique()
                            )
                            == 2
                        ):
                            binary_classification = True
                        else:
                            binary_classification = False
                    else:
                        label_encoder = None
                    # Extract feature names after preprocessing (d) (important if feature selection was performed)
                    available_features = _get_feature_names_after_preprocessing(
                        models_trained_list[selectbox_model_to_be_plotted_int],
                        includes_model=True,
                    )
                    # Download bottons for models and label encoder
                    st.download_button(
                        label="Download model with its pipeline",
                        data=pickle.dumps(
                            models_trained_list[selectbox_model_to_be_plotted]
                        ),
                        file_name="pipeline_"
                        + str(
                            scores_test_df["model"].iloc[selectbox_model_to_be_plotted]
                        ),
                        key="download_model_int",
                        use_container_width=True,
                    )
                    if st.session_state.cv_instance.operation == "classification":
                        st.download_button(
                            label="Download label encoder",
                            data=pickle.dumps(label_encoder),
                            file_name="label_encoder_"
                            + str(
                                scores_test_df["model"].iloc[
                                    selectbox_model_to_be_plotted
                                ]
                            ),
                            key="download_label_encoder_int",
                            use_container_width=True,
                        )

                # Column 2: Display the available plots and plot them
                with col_ip_1_2:
                    # Create three tabs
                    tab_ip_1, tab_ip_2, tab_ip_3 = st.tabs(
                        [
                            "**Partial Dependence Plot**",
                            "**SHAP**",
                            "**Double/Debiased ML**",
                        ]
                    )
                    # Tab ip 1: Partial Dependence Plot
                    with tab_ip_1:
                        # Instantiate placeholders | Session state variables
                        if "fig_pdp" not in st.session_state:
                            st.session_state.fig_pdp = None
                        # Create two columns: to select a feature and to plot it
                        col_ip_2_1, col_ip_2_2 = st.columns([1, 1])
                        with col_ip_2_1:
                            selectbox_feature_to_be_plotted = st.multiselect(
                                label="**Select the feature to be plotted**",
                                options=available_features,
                                default=available_features,
                            )
                        with col_ip_2_2:
                            selectbox_frac_ice = st.selectbox(
                                label="**Select the proportion of the dataset used to plot ICE**",
                                options=[
                                    np.round(x, decimals=2)
                                    for x in np.arange(0.0, 1.1, 0.1)
                                ],
                                index=2,
                            )
                        button_pdp = st.button(
                            label="**Plot**", type="secondary", use_container_width=True
                        )
                        if button_pdp:
                            with st.spinner(
                                "Generating the Partial Dependence Plots..."
                            ):
                                st.session_state.fig_pdp = partial_dependence_plot(
                                    feature=selectbox_feature_to_be_plotted,
                                    pipeline=models_trained_list[
                                        selectbox_model_to_be_plotted_int
                                    ],
                                    X=X_train,
                                    frac_ice=selectbox_frac_ice,
                                )
                            st.success("Done!")
                        if st.session_state.fig_pdp is not None:
                            st.plotly_chart(
                                st.session_state.fig_pdp,
                                theme="streamlit",
                                use_container_width=True,
                            )
                    # Tab ip 2: SHAP
                    with tab_ip_2:
                        # Create two columns
                        col_ip_2_1_1, col_ip_2_1_2 = st.columns([1, 1])
                        with col_ip_2_1_1:
                            selectbox_data_to_be_plotted = st.selectbox(
                                label="**Select the data to be plotted**",
                                options=["test", "train", "complete"],
                                index=0,
                            )
                            # Create variables that contain the data to be plotted
                            if selectbox_data_to_be_plotted == "test":
                                X_to_be_plotted = X_test
                                y_to_be_plotted = y_test
                            elif selectbox_data_to_be_plotted == "train":
                                X_to_be_plotted = X_train
                                y_to_be_plotted = y_train
                            else:
                                X_to_be_plotted = (
                                    st.session_state.cv_instance.data.drop(
                                        [st.session_state.cv_instance.target_variable],
                                        axis=1,
                                    )
                                )
                                y_to_be_plotted = st.session_state.cv_instance.data[
                                    st.session_state.cv_instance.target_variable
                                ]
                            if (
                                st.session_state.cv_instance.operation
                                == "classification"
                            ):
                                y_to_be_plotted = label_encoder.inverse_transform(
                                    y_to_be_plotted
                                )
                                y_to_be_plotted = pd.Series(
                                    y_to_be_plotted,
                                    name=st.session_state.cv_instance.target_variable,
                                )
                        # Create button to compute SHAP values
                        treebased_models = (
                            "DecisionTree",
                            "RandomForest",
                            "ExtraTrees",
                            "CatBoost",
                            "LightGBM",
                            "XGB",
                        )
                        if (
                            scores_test_df["model"]
                            .iloc[selectbox_model_to_be_plotted_int]
                            .startswith(treebased_models)
                        ):
                            button_compute_shap = st.button(
                                label="**Compute SHAP values**",
                                type="secondary",
                                use_container_width=True,
                            )
                        else:
                            with col_ip_2_1_2:
                                st.warning(
                                    "**Caution: The model-agnostic Permutation Explainer will be used, which takes significantly longer than the TreeExplainer**"  # noqa: E501
                                )
                                if len(X_to_be_plotted) > 100:
                                    selectbox_n_samples = st.selectbox(
                                        label="**Select the number of samples to be used to compute SHAP values**",
                                        options=range(100, len(X_to_be_plotted) + 1),
                                        index=0,
                                    )
                                else:
                                    selectbox_n_samples = len(X_to_be_plotted)
                            button_compute_shap = st.button(
                                label="**Compute model-agnostic SHAP values**",
                                type="secondary",
                                use_container_width=True,
                            )
                        # Instantiate placeholders | Session state variables
                        if "shap_values_instance" not in st.session_state:
                            st.session_state.shap_values_instance = None
                        if button_compute_shap:
                            with st.spinner("Computing SHAP values..."):
                                # If the selected model is a treebased model
                                if (
                                    scores_test_df["model"]
                                    .iloc[selectbox_model_to_be_plotted_int]
                                    .startswith(treebased_models)
                                ):
                                    st.session_state.shap_values_instance = (
                                        compute_shap_values_tree(
                                            pipeline=models_trained_list[
                                                selectbox_model_to_be_plotted_int
                                            ],
                                            X=X_to_be_plotted,
                                        )
                                    )
                                else:
                                    st.session_state.shap_values_instance = compute_shap_values_agnostic(
                                        pipeline=models_trained_list[
                                            selectbox_model_to_be_plotted_int
                                        ],
                                        X=X_to_be_plotted,
                                        n_samples=selectbox_n_samples,
                                        operation=st.session_state.cv_instance.operation,
                                    )
                            st.success("Done!")
                        # Once the SHAP values are computed, show download buttons for SHAP values
                        if st.session_state.shap_values_instance is not None:
                            # Create two columns: One to select the plot type, another for the plotting options
                            col_ip_2_2_1, col_ip_2_2_2 = st.columns([1, 1])
                            with col_ip_2_2_1:
                                if len(available_features) < 2:
                                    select_plot = st.radio(
                                        "**Select the plot**",
                                        options=[
                                            "Beeswarm",
                                            "Scatter",
                                        ],
                                    )
                                else:
                                    select_plot = st.radio(
                                        "**Select the plot**",
                                        options=[
                                            "Feature Importance",
                                            "Feature Clustering",
                                            "Beeswarm",
                                            "Scatter",
                                            "Local Explanation",
                                        ],
                                    )
                            with col_ip_2_2_2:
                                # Create a DataFrame with Shapley Values to be plotted (multiclass, binary, regression)
                                if (
                                    len(
                                        st.session_state.shap_values_instance.values.shape
                                    )
                                    > 2
                                ):
                                    if (
                                        st.session_state.shap_values_instance.values.shape[
                                            2
                                        ]
                                        == 2
                                    ):
                                        sv_to_be_plotted = st.session_state.shap_values_instance.values[
                                            :, :, 0
                                        ]
                                    elif (
                                        st.session_state.shap_values_instance.values.shape[
                                            2
                                        ]
                                        > 2
                                    ):
                                        st.markdown(
                                            ":red[**This was a multiclass classification task**]"
                                        )
                                        select_var = st.radio(
                                            "**Select the class of the target variable for which to generate plots.**",
                                            options=label_encoder.classes_,
                                        )
                                        sv_to_be_plotted = st.session_state.shap_values_instance.values[
                                            :,
                                            :,
                                            np.where(
                                                label_encoder.classes_ == select_var
                                            )[0][0],
                                        ]
                                else:
                                    sv_to_be_plotted = (
                                        st.session_state.shap_values_instance.values
                                    )
                                sv_to_be_plotted = pd.DataFrame(
                                    sv_to_be_plotted,
                                    columns=st.session_state.shap_values_instance.feature_names,
                                )
                                # Create selectbox for plotting options
                                selectbox_color = st.selectbox(
                                    label="**Select a color scale**",
                                    options=AVAILABLE_COLORS_SEQUENTIAL,
                                    index=0,
                                    key="tab_ip_2_color",
                                )
                                if select_plot == "Scatter":
                                    # Get list of available features: sorted by importance
                                    available_features = (
                                        sv_to_be_plotted.abs()
                                        .mean(axis=0)
                                        .sort_values(ascending=False)
                                        .index
                                    )
                                    selectbox_feature_to_be_plotted = st.selectbox(
                                        label="**Select the feature to be plotted**",
                                        options=available_features,
                                        index=0,
                                        key="tab_ip_2_features",
                                    )
                                    available_for_color = [
                                        value
                                        for value in available_features
                                        if value not in selectbox_feature_to_be_plotted
                                    ]
                                    selectbox_feature_color = st.selectbox(
                                        label="**Select the variable that determines the color of the bubbles**",
                                        options=[None] + available_for_color,
                                        index=0,
                                    )
                                elif select_plot == "Local Explanation":
                                    selectbox_id = st.selectbox(
                                        label="**Select the id number for which to compute Local Explanations**",
                                        options=range(len(X_to_be_plotted)),
                                        index=0,
                                    )

                            button_plot_shap = st.button(
                                label="**Plot**",
                                type="secondary",
                                use_container_width=True,
                                key="tab_ip_2_plot",
                            )
                            # Create two columns for download buttons
                            col_ip_2_3_1, col_ip_2_3_2 = st.columns([1, 2])
                            with col_ip_2_3_1:
                                st.download_button(
                                    label="Download SHAP values as CSV",
                                    data=convert_dataframe_to_csv(sv_to_be_plotted),
                                    file_name="SHAP_values.csv",
                                    mime="text/csv'",
                                )
                            with col_ip_2_3_2:
                                st.download_button(
                                    label="Download SHAP values as XLSX",
                                    data=convert_dataframe_to_xlsx(sv_to_be_plotted),
                                    file_name="SHAP_values.xlsx",
                                    mime="application/vnd.ms-excel",
                                )
                            # Create title of the plot and instantiate placeholders | Session state variables
                            # scores_test_df['model'].iloc[selectbox_model_to_be_plotted_int]
                            string_to_be_displayed = (
                                str(
                                    scores_test_df["model"].iloc[
                                        selectbox_model_to_be_plotted_int
                                    ]
                                )
                                + " using the "
                                + str(selectbox_data_to_be_plotted)
                                + " data"
                            )
                            # Alternative: string_to_be_displayed = str(models_trained_list[selectbox_model_to_be_plotted_int][-1]) + ' using the ' + str(selectbox_data_to_be_plotted) + ' data'  # noqa: E501
                            if "fig_shap" not in st.session_state:
                                if len(available_features) < 2:
                                    X_prep_bee = pd.DataFrame(
                                        st.session_state.shap_values_instance.data,
                                        columns=st.session_state.shap_values_instance.feature_names,
                                    )
                                    st.session_state.fig_shap = plot_shap_beeswarm(
                                        sv_to_be_plotted,
                                        X=X_prep_bee,
                                        color=selectbox_color,
                                    )
                                else:
                                    st.session_state.fig_shap = (
                                        plot_shap_feature_importance(
                                            sv_to_be_plotted, color=selectbox_color
                                        )
                                    )
                                st.session_state.fig_shap.update_layout(
                                    title=string_to_be_displayed
                                )
                            # Compute the plots
                            if button_plot_shap & (select_plot == "Feature Importance"):
                                st.session_state.fig_shap = (
                                    plot_shap_feature_importance(
                                        sv_to_be_plotted, color=selectbox_color
                                    )
                                )
                                st.session_state.fig_shap.update_layout(
                                    title=string_to_be_displayed
                                )
                            elif button_plot_shap & (
                                select_plot == "Feature Clustering"
                            ):
                                st.session_state.fig_shap = (
                                    plot_shap_feature_clustering(
                                        sv_to_be_plotted, color=selectbox_color
                                    )
                                )
                                st.session_state.fig_shap.update_layout(
                                    title=string_to_be_displayed
                                )
                            elif button_plot_shap & (select_plot == "Beeswarm"):
                                X_prep_bee = pd.DataFrame(
                                    st.session_state.shap_values_instance.data,
                                    columns=st.session_state.shap_values_instance.feature_names,
                                )
                                st.session_state.fig_shap = plot_shap_beeswarm(
                                    sv_to_be_plotted,
                                    X=X_prep_bee,
                                    color=selectbox_color,
                                )
                                st.session_state.fig_shap.update_layout(
                                    title=string_to_be_displayed
                                )
                            elif button_plot_shap & (select_plot == "Scatter"):
                                X_prep_scatter = pd.DataFrame(
                                    st.session_state.shap_values_instance.data,
                                    columns=st.session_state.shap_values_instance.feature_names,
                                )
                                st.session_state.fig_shap = plot_shap_scatter(
                                    shap_values=sv_to_be_plotted,
                                    X=X_prep_scatter,
                                    feature=selectbox_feature_to_be_plotted,
                                    color=selectbox_color,
                                    var_color=selectbox_feature_color,
                                )
                                st.session_state.fig_shap.update_layout(
                                    title=string_to_be_displayed
                                )
                            elif button_plot_shap & (
                                select_plot == "Local Explanation"
                            ):
                                st.session_state.fig_shap = plot_shap_local(
                                    sv_to_be_plotted,
                                    id=selectbox_id,
                                    color=selectbox_color,
                                )
                                st.session_state.fig_shap.update_layout(
                                    title=string_to_be_displayed
                                )
                            # Display the plot
                            #                            st.markdown(f"""**{str(string_to_be_displayed)}**""")
                            st.plotly_chart(
                                st.session_state.fig_shap,
                                theme="streamlit",
                                use_container_width=True,
                            )
                    # Tab ip 3: Double/Debiased ML
                    with tab_ip_3:
                        if len(available_features) < 2:
                            st.warning(
                                "Double/Debiased ML is only available if the number of predictors is greater than"
                                " one.  \n"
                                " **For more information of Double/Debiased ML, see:** Chernozhukov, V., Chetverikov,"
                                " D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018)."
                                " Double/debiased machine learning for treatment and structural parameters."
                                " The Econometrics Journal, 21(1), C1-C68. https://doi.org/10.1111/ectj.12097"
                            )
                        else:
                            # Instantiate placeholders | Session state variables
                            if "ate_df" not in st.session_state:
                                st.session_state.ate_df = None
                            button_d_d_ml = st.button(
                                label="**Compute ATE**",
                                type="secondary",
                                use_container_width=True,
                                help="**For more information of Double/Debiased ML, see:** Chernozhukov,"
                                " V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., &"
                                " Robins, J. (2018). Double/debiased machine learning for treatment and"
                                " structural parameters. The Econometrics Journal, 21(1), C1-C68."
                                " https://doi.org/10.1111/ectj.12097",
                            )
                            with st.spinner("Computing ATE..."):
                                if button_d_d_ml:
                                    st.session_state.ate_df = compute_average_treatment_effect(
                                        pipeline=models_trained_list[
                                            selectbox_model_to_be_plotted
                                        ],
                                        data=data,
                                        target_variable=st.session_state.cv_instance.target_variable,
                                        estimation_method="custom",
                                        operation=st.session_state.cv_instance.operation,
                                        label_encoder=label_encoder,
                                    )
                                    st.success("Done!")
                            if st.session_state.ate_df is not None:
                                # Create two columns for download buttons
                                col_ip_3_1, col_ip_3_2 = st.columns([1, 2])
                                with col_ip_3_1:
                                    st.download_button(
                                        label="Download ATE values as CSV",
                                        data=convert_dataframe_to_csv(
                                            st.session_state.ate_df
                                        ),
                                        file_name="ATE_values.csv",
                                        mime="text/csv'",
                                    )
                                with col_ip_3_2:
                                    st.download_button(
                                        label="Download ATE values as XLSX",
                                        data=convert_dataframe_to_xlsx(
                                            st.session_state.ate_df
                                        ),
                                        file_name="ATE_values.xlsx",
                                        mime="application/vnd.ms-excel",
                                    )
                                st.session_state.fig_ate = plot_ate(
                                    st.session_state.ate_df
                                )
                                # Create two columns: to display the DataFrame and the Plot
                                col_ip_3_1, col_ip_3_2 = st.columns([1, 1])
                                with col_ip_3_1:
                                    st.dataframe(
                                        st.session_state.ate_df,
                                        use_container_width=True,
                                    )
                                with col_ip_3_2:
                                    st.plotly_chart(
                                        st.session_state.fig_ate,
                                        theme="streamlit",
                                        use_container_width=True,
                                    )


#    streamlit_profiler.stop()


if __name__ == "__main__":
    # Page setup
    st.set_page_config(
        page_title="DataChum", page_icon="assets/logo_01.png", layout="wide"
    )
    # Create file uploader object
    uploaded_file = st.file_uploader("Upload your database", type=["csv", "xlsx"])
    # Set placeholder for data
    if "data" not in st.session_state:
        st.session_state.data = None
    if uploaded_file is not None:
        # Read the file to a dataframe using pandas
        if uploaded_file.name[-3:] == "csv":
            # Read in the csv file
            st.session_state.data = read_csv(uploaded_file)
        elif uploaded_file.name[-4:] == "xlsx":
            # Read in the csv file
            st.session_state.data = read_xlsx(uploaded_file)
        else:
            st.write("Type should be .CSV or .XLSX")
    main()
