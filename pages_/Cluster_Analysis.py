# Import moduls from local directories
from assets.colors import AVAILABLE_COLORS_SEQUENTIAL
from modules.classification_and_regression.evaluation import corrected_repeated_t_test
from modules.classification_and_regression.models import AVAILABLE_MODELS_CLASSIFICATION
from modules.cluster.evaluation import (
    get_cluster_labels_and_X_prep,
    line_plot,
    prepare_results_for_line_plot_metrics,
    prepare_results_for_line_plot_models,
    silhouette_plot,
)
from modules.cluster.main import clustering, clustering_cross_validation
from modules.cluster.metrics import (
    AVAILABLE_METRICS_TO_MONITOR_CLUSTERING_CONVERGENCE,
    AVAILABLE_METRICS_TO_MONITOR_CLUSTERING_CROSSVALIDATION_CONVERGENCE,
)
from modules.exploratory_data_analysis.univariate_and_bivariate import plot_num
from modules.utils.load_and_save_data import (
    convert_dataframe_to_csv,
    convert_dataframe_to_xlsx,
    read_csv,
    read_xlsx,
)
from modules.utils.preprocessing import (
    clean_strings_and_feature_names,
    AVAILABLE_IMPUTATION_CATEGORICAL,
    AVAILABLE_IMPUTATION_NUMERICAL,
    AVAILABLE_SCALER,
)
from pages_ import Exploratory_Data_Analysis

# Import the required libraries
import pandas as pd
import streamlit as st

# from streamlit_profiler import Profiler


def main():
    st.subheader("Cluster Analysis")

    # Profile the app
    #    streamlit_profiler = Profiler()
    #    streamlit_profiler.start()

    # Copy data from session state
    if st.session_state.data is not None:
        data = st.session_state.data
        # Clean strings and feature names
        data = clean_strings_and_feature_names(data=data)
        # Drop ID columns (or similar): Analyze whether all values of the column are unique
        # (count of unique values equals column's length)
        cols_all = data.columns.to_list()
        list_of_dropped_columns = []
        for column in cols_all:
            if len(data[column]) == len(data[column].unique()):
                data = data.drop(column, axis=1)
                list_of_dropped_columns.append(column)
        if len(list_of_dropped_columns) > 0:
            #            st.markdown(f"""**{str(string_to_be_displayed)}**""")
            st.warning(
                "Following columns have been removed as all values of the column are unique: "
                + ", ".join(list_of_dropped_columns)
            )
        # Get column names (also NUMERICAL and CATEGORICAL)
        cols_num = data.select_dtypes(include=["float", "int"]).columns.to_list()
        cols_cat = data.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.to_list()
        # Create three columns
        col_1, col_2, col_3 = st.columns(3)
        with col_1:
            st.markdown(
                "Please check if the data types of the features were infered correctly (Integers will be handled as numerical variables)"  # noqa: E501
            )
        with col_2:
            cols_cat_df = pd.DataFrame({"Categorical Features": cols_cat})
            st.dataframe(cols_cat_df, use_container_width=True)
        with col_3:
            cols_num_df = pd.DataFrame({"Numerical Features": cols_num})
            st.dataframe(cols_num_df, use_container_width=True)
        st.subheader("**Cluster Modeling Setup**")
        # Create three tabs
        tab_s1, tab_s2, tab_s3 = st.tabs(
            [
                "**Procedure**",
                "**Preprocessing**",
                "**Models**",
            ]
        )
        # Tab 1: Procedure
        with tab_s1:
            col_1, col_2, col_3 = st.columns([1.0, 1.5, 1.5])
            with col_1:
                selectradio_procedure = st.radio(
                    label="**Select the procedure**",
                    options=[
                        "Standart Cluster Analysis with optional bootstrapping",
                        "Cluster Analysis using prediction-based k-fold cross-validation method",
                    ],
                    index=1,
                    help="**For more information of the prediction-based resampling method, see:** Dudoit, S.,"
                    " & Fridlyand, J. (2002). A prediction-based resampling method for estimating the number"
                    " of clusters in a dataset. Genome Biology, 3(7), 1-21.",
                )
            with col_2:
                selectbox_n_cluster_min = st.selectbox(
                    label="**Select the minimum number of clusters**",
                    options=range(2, len(data)),
                    index=0,
                )
                selectbox_n_cluster_max = st.selectbox(
                    label="**Select the maximum number of clusters**",
                    options=range(selectbox_n_cluster_min, len(data) + 1),
                    index=0,
                )
                range_min_max_clusters = (
                    selectbox_n_cluster_max - selectbox_n_cluster_min
                )
                if range_min_max_clusters > 3:
                    selectbox_n_consecutive_clusters_without_improvement = st.selectbox(
                        label="**Monitor convergence of adding clusters: Select the maximum number of consecutive "
                        "attempts without improvement in the evaluation metric**",
                        options=range(1, range_min_max_clusters),
                        index=2,
                    )
                else:
                    selectbox_n_consecutive_clusters_without_improvement = (
                        selectbox_n_cluster_max
                    )
            with col_3:
                # Selectboxes for bootstrap samples and/or fold numbers
                if (
                    selectradio_procedure
                    == "Standart Cluster Analysis with optional bootstrapping"
                ):
                    selectbox_n_bootstrap_samples = st.selectbox(
                        label="**Select the number of bootstrap samples**",
                        options=range(0, 5001, 50),
                        index=0,
                    )
                    if selectbox_n_bootstrap_samples > 200:
                        selectbox_n_consecutive_bootstraps_without_improvement = st.selectbox(
                            label="**Every 50 bootstrap replicates, monitor convergence: Select the maximum number "
                            "of consecutive attempts without improvement in the coeficient of variation**",
                            options=range(1, int(selectbox_n_bootstrap_samples / 50)),
                            index=2,
                        )
                    else:
                        selectbox_n_consecutive_bootstraps_without_improvement = (
                            selectbox_n_bootstrap_samples
                        )
                    # Selectboxes for evaluation metrics
                    if (range_min_max_clusters > 3) | (
                        selectbox_n_bootstrap_samples > 200
                    ):
                        selectbox_monitor_metric = st.selectbox(
                            label="**Select the evaluation metric to monitor convergence**",
                            options=AVAILABLE_METRICS_TO_MONITOR_CLUSTERING_CONVERGENCE,
                            index=2,
                        )
                    # Create a placeholder for selectbox_monitor_metric
                    else:
                        selectbox_monitor_metric = "Silhouette"
                # If Cluster Analysis using prediction-based resampling method
                else:
                    selectbox_n_inner_cv_reps = st.selectbox(
                        label="**Select the number of times cross-validator needs to be repeated**",
                        options=range(1, 11, 1),
                        index=4,
                    )
                    selectbox_n_inner_cv_folds = st.selectbox(
                        label="**Select the number of folds**",
                        options=range(5, 11, 1),
                        index=5,
                    )
                    selectbox_classification_model = st.selectbox(
                        label="**Select the classification model to use for predictions**",
                        options=AVAILABLE_MODELS_CLASSIFICATION,
                        index=0,
                    )
                    # Selectboxes for evaluation metrics
                    if range_min_max_clusters > 3:
                        selectbox_monitor_metric = st.selectbox(
                            label="**Select the evaluation metric to monitor convergence**",
                            options=AVAILABLE_METRICS_TO_MONITOR_CLUSTERING_CROSSVALIDATION_CONVERGENCE,
                            index=2,
                        )
                    # Create a placeholder for selectbox_monitor_metric
                    else:
                        selectbox_monitor_metric = "Silhouette"
        # Tab 2: Preprocessing
        with tab_s2:
            col_1, col_2 = st.columns(2)
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
        # Tab 3: Models
        with tab_s3:
            models_to_evaluate = []
            col_1, col_2, col_3, col_4 = st.columns(4)
            with col_1:
                checkbox_1 = st.checkbox("AgglomerativeClustering_average")
                if checkbox_1:
                    models_to_evaluate.append("AgglomerativeClustering_average")
                checkbox_2 = st.checkbox("AgglomerativeClustering_complete")
                if checkbox_2:
                    models_to_evaluate.append("AgglomerativeClustering_complete")
                checkbox_3 = st.checkbox("AgglomerativeClustering_single")
                if checkbox_3:
                    models_to_evaluate.append("AgglomerativeClustering_single")
                checkbox_4 = st.checkbox("AgglomerativeClustering_ward", value=True)
                if checkbox_4:
                    models_to_evaluate.append("AgglomerativeClustering_ward")
            with col_2:
                checkbox_5 = st.checkbox("BayesianGaussianMixture_diag")
                if checkbox_5:
                    models_to_evaluate.append("BayesianGaussianMixture_diag")
                checkbox_6 = st.checkbox("BayesianGaussianMixture_full")
                if checkbox_6:
                    models_to_evaluate.append("BayesianGaussianMixture_full")
                checkbox_7 = st.checkbox("BayesianGaussianMixture_spherical")
                if checkbox_7:
                    models_to_evaluate.append("BayesianGaussianMixture_spherical")
                checkbox_8 = st.checkbox("BayesianGaussianMixture_tied")
                if checkbox_8:
                    models_to_evaluate.append("BayesianGaussianMixture_tied")
            with col_3:
                checkbox_9 = st.checkbox("GaussianMixture_diag")
                if checkbox_9:
                    models_to_evaluate.append("GaussianMixture_diag")
                checkbox_10 = st.checkbox("GaussianMixture_full")
                if checkbox_10:
                    models_to_evaluate.append("GaussianMixture_full")
                checkbox_11 = st.checkbox("GaussianMixture_spherical")
                if checkbox_11:
                    models_to_evaluate.append("GaussianMixture_spherical")
                checkbox_12 = st.checkbox("GaussianMixture_tied")
                if checkbox_12:
                    models_to_evaluate.append("GaussianMixture_tied")
            with col_4:
                checkbox_13 = st.checkbox(
                    "SpectralClustering_nearest_neighbors", value=True
                )
                if checkbox_13:
                    models_to_evaluate.append("SpectralClustering_nearest_neighbors")
                checkbox_14 = st.checkbox("SpectralClustering_rbf")
                if checkbox_14:
                    models_to_evaluate.append("SpectralClustering_rbf")
                checkbox_15 = st.checkbox("BIRCH")
                if checkbox_15:
                    models_to_evaluate.append("BIRCH")
                checkbox_16 = st.checkbox("KMeans")
                if checkbox_16:
                    models_to_evaluate.append("KMeans")
                # If standart procedure, add "DBSCAN" model (cannot be used with prediction-based)
                if (
                    selectradio_procedure
                    == "Standart Cluster Analysis with optional bootstrapping"
                ):
                    checkbox_17 = st.checkbox("DBSCAN")
                    if checkbox_17:
                        models_to_evaluate.append("DBSCAN")
        # Instantiate placeholders | Session state variables
        if "cluster_instance" not in st.session_state:
            st.session_state.cluster_instance = None
        # Create a button to run the cross-validation
        button_run_cv = st.button(
            label="**Run cluster analysis**", type="primary", use_container_width=True
        )
        if button_run_cv:
            with st.spinner("Fitting the selected models..."):
                if (
                    selectradio_procedure
                    == "Standart Cluster Analysis with optional bootstrapping"
                ):
                    st.session_state.cluster_instance = clustering(
                        data=data,
                        imputation_numerical=selectbox_imput_num,
                        imputation_categorical=selectbox_imput_cat,
                        scaler=selectbox_scaler,
                        cluster_models=models_to_evaluate,
                        n_cluster_min=selectbox_n_cluster_min,
                        n_cluster_max=selectbox_n_cluster_max,
                        n_bootstrap_samples=selectbox_n_bootstrap_samples,
                        n_consecutive_bootstraps_without_improvement=selectbox_n_consecutive_bootstraps_without_improvement,  # noqa: E501
                        n_consecutive_clusters_without_improvement=selectbox_n_consecutive_clusters_without_improvement,
                        monitor_metric=selectbox_monitor_metric,
                    )
                else:
                    st.session_state.cluster_instance = clustering_cross_validation(
                        data=data,
                        imputation_numerical=selectbox_imput_num,
                        imputation_categorical=selectbox_imput_cat,
                        scaler=selectbox_scaler,
                        cluster_models=models_to_evaluate,
                        n_cluster_min=selectbox_n_cluster_min,
                        n_cluster_max=selectbox_n_cluster_max,
                        classification_model=[selectbox_classification_model],
                        inner_cv_folds=selectbox_n_inner_cv_folds,
                        inner_cv_rep=selectbox_n_inner_cv_reps,
                        n_consecutive_clusters_without_improvement=selectbox_n_consecutive_clusters_without_improvement,
                        monitor_metric=selectbox_monitor_metric,
                    )
            st.success("Done!")
            # Rerun app (tab hack: prevent jumping back to first tab)
            # https://github.com/streamlit/streamlit/issues/4996#issuecomment-1261464494
            st.experimental_rerun()

        if st.session_state.cluster_instance is not None:
            st.subheader("**Model Evaluation**")
            # Create three tabs
            tab_e1_1, tab_e1_2, tab_e1_3 = st.tabs(
                [
                    "**Clustering Scores**",
                    "**Evaluation Plots**",
                    "**Interpretation Plots**",
                ]
            )
            # Tab 1: Clustering Scores
            with tab_e1_1:
                # Create DataFrame with clustering scores
                scores_df = st.session_state.cluster_instance.all_results
                # Create two columns for the download buttons
                col_cluster_score_1_1, col_cluster_score_1_2 = st.columns([1, 2])
                with col_cluster_score_1_1:
                    st.download_button(
                        label="Download clustering scores as CSV",
                        data=convert_dataframe_to_csv(scores_df),
                        file_name="clustering_scores.csv",
                        mime="text/csv'",
                    )
                with col_cluster_score_1_2:
                    st.download_button(
                        label="Download clustering scores as XLSX",
                        data=convert_dataframe_to_xlsx(scores_df),
                        file_name="clustering_scores.xlsx",
                        mime="application/vnd.ms-excel",
                    )
                # Create tabs to display results and to compute t-tests
                col_cluster_score_2_1, col_cluster_score_2_2 = st.columns([2.0, 1.0])
                # Display cross-validation results
                with col_cluster_score_2_1:
                    st.markdown("**Grouped by model & number of clusters**")
                    scores_df_grouped = (
                        scores_df.groupby(by=["model", "n_clusters"])
                        .mean()
                        .reset_index(drop=False)
                    )
                    st.dataframe(
                        scores_df_grouped.style.format(precision=3),
                        height=((len(scores_df_grouped) + 1) * 35 + 3),
                        use_container_width=False,
                    )
                    st.markdown("**Complete**")
                    st.dataframe(
                        scores_df.set_index(["model", "n_clusters"]).style.format(
                            "{:.3f}"
                        ),
                        use_container_width=False,
                    )
                # Compute and display t-test results
                idx_models_to_be_compared = range(len(scores_df_grouped))
                with col_cluster_score_2_2:
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
                    if st.session_state.cluster_instance.method == "standart":
                        st.markdown(
                            "Only available for Cluster Analysis using prediction-based k-fold cross-validation method"
                            " & if two ore more models were evaluated & if the number of k-folds >= 10 and "
                            "the number of repetitions >= 5"
                        )
                    elif (st.session_state.cluster_instance.method == "k-fold") & (
                        (len(scores_df_grouped) < 2)
                        | (st.session_state.cluster_instance.inner_cv_folds < 10)
                        | (st.session_state.cluster_instance.inner_cv_rep < 5)
                    ):
                        st.markdown(
                            "Only available for Cluster Analysis using prediction-based k-fold cross-validation method"
                            " & if two ore more models were evaluated & if the number of k-folds >= 10 and "
                            "the number of repetitions >= 5"
                        )
                    else:
                        selectbox_t_test_model_1 = st.selectbox(
                            label="**Select the index number of model 1 to be compared**",
                            options=idx_models_to_be_compared,
                            index=0,
                            key="t_test_cluster_model_1",
                        )
                        selectbox_t_test_model_2 = st.selectbox(
                            label="**Select the index number of model 2 to be compared**",
                            options=[
                                value
                                for value in idx_models_to_be_compared
                                if value != selectbox_t_test_model_1
                            ],
                            index=0,
                            key="t_test_cluster_model_2",
                        )
                        selectbox_t_test_evaluation_metric = st.selectbox(
                            label="**Select the evaluation metric to be compared**",
                            options=[
                                value
                                for value in scores_df.columns.to_list()
                                if value not in ["model", "n_clusters"]
                            ],
                            index=0,
                            key="t_test_cluster_evaluation_metric",
                        )
                        # Extract scores to compute corrected_repeated_t_test
                        scores_model_1 = scores_df[
                            (
                                scores_df["model"]
                                == scores_df_grouped.iloc[selectbox_t_test_model_1, :][
                                    "model"
                                ]
                            )
                            & (
                                scores_df["n_clusters"]
                                == scores_df_grouped.iloc[selectbox_t_test_model_1, :][
                                    "n_clusters"
                                ]
                            )
                        ][selectbox_t_test_evaluation_metric]
                        scores_model_2 = scores_df[
                            (
                                scores_df["model"]
                                == scores_df_grouped.iloc[selectbox_t_test_model_2, :][
                                    "model"
                                ]
                            )
                            & (
                                scores_df["n_clusters"]
                                == scores_df_grouped.iloc[selectbox_t_test_model_2, :][
                                    "n_clusters"
                                ]
                            )
                        ][selectbox_t_test_evaluation_metric]
                        # Compute t-test
                        result_t_test = corrected_repeated_t_test(
                            scores_model_1=scores_model_1,
                            scores_model_2=scores_model_2,
                            n_folds=st.session_state.cluster_instance.inner_cv_folds,
                            n=len(data),
                        )
                        # Change model names
                        result_t_test.result_descriptives["model"] = [
                            selectbox_t_test_model_1,
                            selectbox_t_test_model_2,
                        ]
                        st.markdown(
                            f"""**Descriptives - {str(selectbox_t_test_evaluation_metric)}**"""
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
            # Tab 2: Evaluation Plaots
            with tab_e1_2:
                # Create uo to three tabs: Bar Plots, Silhouette Plot, Line Plots
                if len(scores_df_grouped["n_clusters"].unique()) < 2:
                    # Create two tabs
                    tab_e1_2_1, tab_e1_2_2 = st.tabs(
                        [
                            "**Bar Plots**",
                            "**Silhouette Plots**",
                        ]
                    )
                else:
                    tab_e1_2_1, tab_e1_2_2, tab_e1_2_3 = st.tabs(
                        [
                            "**Bar Plots**",
                            "**Silhouette Plots**",
                            "**Line Plots**",
                        ]
                    )
                # Bar Plots
                with tab_e1_2_1:
                    if len(scores_df_grouped) < 2:
                        var_cat = None
                        data_to_be_plotted = scores_df
                        multiple = False
                    elif len(scores_df_grouped["n_clusters"].unique()) < 2:
                        var_cat = "model"
                        data_to_be_plotted = scores_df
                        multiple = False
                    elif len(scores_df_grouped["model"].unique()) < 2:
                        var_cat = "n_clusters"
                        data_to_be_plotted = scores_df
                        multiple = False
                    else:
                        multiple = True
                    # Create two columns for plotting options
                    (
                        col_box_1,
                        col_box_2,
                    ) = st.columns([1, 3])
                    with col_box_1:
                        if multiple is True:
                            selectbox_x_axis = st.selectbox(
                                label="**Select the variable to be plotted on x-axis**",
                                options=["n_clusters", "model"],
                                index=0,
                                key="boxplot_1_xaxis",
                            )
                            if selectbox_x_axis == "model":
                                selectbox_n_clusters = st.selectbox(
                                    label="**Select the cluster solution to be plotted**",
                                    options=scores_df_grouped["n_clusters"].unique(),
                                    index=0,
                                    key="boxplot_1_n_clusters",
                                )
                                var_cat = "model"
                                data_to_be_plotted = scores_df[
                                    scores_df["n_clusters"] == selectbox_n_clusters
                                ]
                            else:
                                selectbox_model_boxplot = st.selectbox(
                                    label="**Select the model to be plotted**",
                                    options=scores_df_grouped["model"].unique(),
                                    index=0,
                                    key="boxplot_1_model",
                                )
                                var_cat = "n_clusters"
                                data_to_be_plotted = scores_df[
                                    scores_df["model"] == selectbox_model_boxplot
                                ]
                        selectbox_boxplot_evaluation_metric = st.selectbox(
                            label="**Select the evaluation metric to be plotted**",
                            options=[
                                value
                                for value in scores_df.columns.to_list()
                                if value not in ["model", "n_clusters"]
                            ],
                            index=0,
                            key="boxplot_1_evaluation_metric",
                        )
                        # Create selectbox for plotting options
                        selectbox_color = st.selectbox(
                            label="**Select a color scale**",
                            options=AVAILABLE_COLORS_SEQUENTIAL,
                            index=0,
                            key="tab_box_1_color",
                        )
                    with col_box_2:
                        fig_variable = plot_num(
                            data=data_to_be_plotted,
                            var_num=selectbox_boxplot_evaluation_metric,
                            var_cat=var_cat,
                            plot_type="Box-Plot",
                            color=selectbox_color,
                            template="plotly_white",
                        )
                        st.plotly_chart(
                            fig_variable,
                            theme="streamlit",
                            use_container_width=True,
                        )
                # Silhouette Plot
                with tab_e1_2_2:
                    st.markdown(
                        "**CAUTION: Generation of plots requires re-training the model.**"
                    )
                    # Instantiate placeholders | Session state variables
                    if "fig_sil_plot" not in st.session_state:
                        st.session_state.fig_sil_plot = None
                    # Create two columns for plotting options
                    col_sil_1, col_sil_2 = st.columns([1, 3])
                    with col_sil_1:
                        selectbox_model_sil_plot = st.selectbox(
                            label="**Select the model to be plotted**",
                            options=scores_df_grouped["model"].unique(),
                            index=0,
                            key="sil_plot_1_model",
                        )
                        selectbox_n_cluster_sil_plot = st.selectbox(
                            label="**Select the cluster solution to be plotted**",
                            options=scores_df_grouped["n_clusters"].unique(),
                            index=0,
                            key="sil_plot_1_n_cluster",
                        )
                        selectbox_color = st.selectbox(
                            label="**Select a color scale**",
                            options=AVAILABLE_COLORS_SEQUENTIAL,
                            index=0,
                            key="tab_sil_1_color",
                        )
                        button_generate_sil_plot = st.button(
                            label="**Plot**",
                            type="secondary",
                            use_container_width=True,
                            key="tab_e1_2_2_plot",
                        )
                    with col_sil_2:
                        if button_generate_sil_plot:
                            cluster_labels, X_prep = get_cluster_labels_and_X_prep(
                                data=data,
                                imputation_numerical=st.session_state.cluster_instance.imputation_numerical,
                                imputation_categorical=st.session_state.cluster_instance.imputation_categorical,
                                scaler=st.session_state.cluster_instance.scaler,
                                cluster_model=[selectbox_model_sil_plot],
                                n_clusters=selectbox_n_cluster_sil_plot,
                            )
                            st.session_state.fig_sil_plot = silhouette_plot(
                                X_prep=X_prep,
                                cluster_model=selectbox_model_sil_plot,
                                cluster_labels=cluster_labels,
                                color=selectbox_color,
                            )
                        if st.session_state.fig_sil_plot is not None:
                            st.plotly_chart(
                                st.session_state.fig_sil_plot,
                                theme="streamlit",
                                use_container_width=True,
                            )
                # Line Plot
                if len(scores_df_grouped["n_clusters"].unique()) >= 2:
                    with tab_e1_2_3:
                        if len(scores_df_grouped["model"].unique()) < 2:
                            data_to_be_plotted = prepare_results_for_line_plot_metrics(
                                scores_df,
                                st.session_state.cluster_instance.cluster_models[0],
                            )
                            y_axis = "scaled_score"
                            traces = "metric"
                            multiple = False
                        else:
                            multiple = True
                        # Create two columns for plotting options
                        (
                            col_line_1,
                            col_line_2,
                        ) = st.columns([1, 3])
                        with col_line_1:
                            if multiple is True:
                                selectbox_var_comp = st.selectbox(
                                    label="**Select the variable to be compared**",
                                    options=["model", "metric"],
                                    index=0,
                                    key="lineplot_1_var_comp",
                                )
                                if selectbox_var_comp == "model":
                                    selectbox_metric_lineplot = st.selectbox(
                                        label="**Select the metric to be plotted**",
                                        options=[
                                            value
                                            for value in scores_df.columns.to_list()
                                            if value not in ["model", "n_clusters"]
                                        ],
                                        index=0,
                                        key="lineplot_1_metric",
                                    )
                                    y_axis = selectbox_metric_lineplot
                                    traces = "model"
                                    data_to_be_plotted = (
                                        prepare_results_for_line_plot_models(scores_df)
                                    )
                                else:
                                    selectbox_model_lineplot = st.selectbox(
                                        label="**Select the model to be plotted**",
                                        options=scores_df_grouped["model"].unique(),
                                        index=0,
                                        key="lineplot_1_model",
                                    )
                                    y_axis = "scaled_score"
                                    traces = "metric"
                                    data_to_be_plotted = (
                                        prepare_results_for_line_plot_metrics(
                                            scores_df,
                                            selectbox_model_lineplot,
                                        )
                                    )
                            # Create selectbox for plotting options
                            selectbox_color = st.selectbox(
                                label="**Select a color scale**",
                                options=AVAILABLE_COLORS_SEQUENTIAL,
                                index=0,
                                key="tab_line_1_color",
                            )
                        with col_line_2:
                            if y_axis == "scaled_score":
                                st.markdown(
                                    "**CAUTION: A MinMax-Scaler (0, 1) was applied to"
                                    " Calinski-Harabasz & Davies-Bouldin scores**"
                                )
                            fig_variable = line_plot(
                                data=data_to_be_plotted,
                                x="n_clusters",
                                y=y_axis,
                                traces=traces,
                                color=selectbox_color,
                            )
                            st.plotly_chart(
                                fig_variable,
                                theme="streamlit",
                                use_container_width=True,
                            )
            # Tab 3: Interpretation Plots
            with tab_e1_3:
                # Create two columns for plotting options
                col_int_1, col_int_2 = st.columns([1, 3])
                with col_int_1:
                    selectbox_model_int_plot = st.selectbox(
                        label="**Select the model to be plotted**",
                        options=scores_df_grouped["model"].unique(),
                        index=0,
                        key="int_plot_1_model",
                    )
                    selectbox_n_cluster_int_plot = st.selectbox(
                        label="**Select the cluster solution to be plotted**",
                        options=scores_df_grouped["n_clusters"].unique(),
                        index=0,
                        key="int_plot_1_n_cluster",
                    )
                    selectbox_color = st.selectbox(
                        label="**Select a color scale**",
                        options=AVAILABLE_COLORS_SEQUENTIAL,
                        index=0,
                        key="tab_int_1_color",
                    )
                    cluster_labels, X_prep = get_cluster_labels_and_X_prep(
                        data=data,
                        imputation_numerical=st.session_state.cluster_instance.imputation_numerical,
                        imputation_categorical=st.session_state.cluster_instance.imputation_categorical,
                        scaler=st.session_state.cluster_instance.scaler,
                        cluster_model=[selectbox_model_int_plot],
                        n_clusters=selectbox_n_cluster_int_plot,
                    )
                    # Create a DataFrames to be plotted (with preprocessed data)
                    data_to_be_plotted = X_prep
                    data_to_be_plotted["cluster_label"] = cluster_labels
                    data_to_be_plotted = (
                        data_to_be_plotted.groupby(by="cluster_label")
                        .mean()
                        .reset_index()
                        .melt(
                            id_vars="cluster_label",
                            value_vars=X_prep.columns.to_list(),
                        )
                    )
                    # Create a DataFrame to be downloaded (with original data)
                    data_to_be_downloaded = data
                    data_to_be_downloaded["cluster_label"] = [
                        "Cluster " + str(value) for value in cluster_labels
                    ]
                with col_int_2:
                    # Create two columns for download buttons
                    col_int_2_1, col_int_2_2, col_int_2_3 = st.columns(3)
                    with col_int_2_1:
                        st.download_button(
                            label="Assign cluster labels to database and download as CSV",
                            data=convert_dataframe_to_csv(data_to_be_downloaded),
                            file_name="data_with_cluster_labels.csv",
                            mime="text/csv'",
                        )
                    with col_int_2_2:
                        st.download_button(
                            label="Assign cluster labels to database and download as XLSX",
                            data=convert_dataframe_to_xlsx(
                                data_to_be_downloaded.astype("object")
                            ),
                            file_name="data_with_cluster_labels.xlsx",
                            mime="application/vnd.ms-excel",
                        )
                    with col_int_2_3:
                        # Assign labels and go to EDA module
                        button_assign_labels = st.button(
                            label="Assign cluster labels to database and go to Exploratory Data Analysis module",
                            type="secondary",
                            use_container_width=True,
                            key="tab_e1_3_assign_labels",
                        )
                        if button_assign_labels:
                            st.session_state.data = data_to_be_downloaded
                            st.session_state.data_updated = True
                            st.session_state.page = Exploratory_Data_Analysis
                    # Show the interpretation plot
                    fig_variable = line_plot(
                        data=data_to_be_plotted,
                        x="variable",
                        y="value",
                        traces="cluster_label",
                        color=selectbox_color,
                    )
                    st.plotly_chart(
                        fig_variable,
                        theme="streamlit",
                        use_container_width=True,
                    )


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
