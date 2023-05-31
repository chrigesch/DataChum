# Import moduls from local directories
from modules.classification_and_regression.models import AVAILABLE_MODELS_CLASSIFICATION
from modules.cluster.metrics import (
    AVAILABLE_METRICS_TO_MONITOR_CLUSTERING_CONVERGENCE,
    AVAILABLE_METRICS_TO_MONITOR_CLUSTERING_CROSSVALIDATION_CONVERGENCE,
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
)

# Import the required libraries
import numpy as np
import pandas as pd
import streamlit as st

# from streamlit_profiler import Profiler


def main():
    st.subheader("Cluster Analysis")

    # Profile the app
    #    streamlit_profiler = Profiler()
    #    streamlit_profiler.start()

    # Create file uploader object
    uploaded_file = st.file_uploader("Upload your database", type=["csv", "xlsx"])

    if uploaded_file is not None:
        # Read the file to a dataframe using pandas
        if uploaded_file.name[-3:] == "csv":
            # Read in the csv file
            data = read_csv(uploaded_file)
        elif uploaded_file.name[-4:] == "xlsx":
            # Read in the csv file
            data = read_xlsx(uploaded_file)
        else:
            st.write("Type should be .CSV or .XLSX")

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
            st.markdown(
                ":red[**Following columns have been removed as all values of the column are unique:**] "
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
                        label="**Monitor convergence of adding clusters: Select the maximum number of consecutive attempts without improvement in the evaluation metric**",
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
                            label="**Every 50 bootstrap replicates, monitor convergence: Select the maximum number of consecutive attempts without improvement in the coeficient of variation**",
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


if __name__ == "__main__":
    # Page setup
    st.set_page_config(
        page_title="DataChum", page_icon="assets/logo_01.png", layout="wide"
    )
    main()
