# Import moduls from local directories
from assets.colors import AVAILABLE_COLORS_DIVERGING, AVAILABLE_COLORS_SEQUENTIAL
from modules.classification_and_regression.cv_workflow import (
    AVAILABLE_SCORES_CLASSIFICATION,
    AVAILABLE_SCORES_REGRESSION,
)
from modules.exploratory_data_analysis.associations import (
    associations_for_categorical_and_numerical_variables,
    plot_heatmap,
)
from modules.exploratory_data_analysis.multivariate import (
    AVAILABLE_MANIFOLD_IMPLEMENTATIONS,
    plot_bubble_chart,
    plot_manifold,
    plot_num_with_grouping_variable,
    plot_pca_2d,
    plot_pca_3d,
    plot_pca_explained_variances,
    plot_random_feature_dropping_curve,
    principal_component_analysis,
)
from modules.exploratory_data_analysis.overview import (
    eda_overview,
    eda_overview_detailed,
)
from modules.exploratory_data_analysis.univariate_and_bivariate import (
    AVAILABLE_TEMPLATES,
    plot_cat,
    plot_num,
    plot_cat_cat,
    plot_num_num,
)
from modules.utils.load_and_save_data import (
    convert_dataframe_to_xlsx,
    convert_dataframe_to_csv,
    read_csv,
    read_xlsx,
)

# Import the required libraries
from math import sqrt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# from streamlit_profiler import Profiler


def main():
    # Add a subtitle
    st.subheader("Exploratory Data Analysis")

    # Profile the app
    #    streamlit_profiler = Profiler()
    #    streamlit_profiler.start()

    # Copy data from session state
    if st.session_state.data is not None:
        data = st.session_state.data
        # Get NUMERICAL, CATEGORICAL and DATETIME column names (Do NOT include DATETIME in "cols_all")
        cols_num = data.select_dtypes(include=["float", "int"]).columns.to_list()
        cols_cat = data.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.to_list()
        # cols_tim = data.select_dtypes(include=["datetime"]).columns.to_list()

        cols_all = data.columns.to_list()
        cols_cat_and_num = data.select_dtypes(exclude=["datetime"]).columns.to_list()

        # Asserts
        assert len(cols_all) > 1, "Database must contain 2 or more columns"
        assert (
            len(cols_cat_and_num) >= 1
        ), "Database must contain 1 or more categorical or numerical columns"
        # Compute associations (necesary for the overview tab)
        associations_df = associations_for_categorical_and_numerical_variables(data)[0]

        # Create tabs, according to the number of CATEGORICAL and NUMERICAL columns
        if len(cols_cat_and_num) == 1:
            tab_1, tab_2, tab_3 = st.tabs(
                ["Overview", "Univariate Analysis", "Missings"]
            )
        elif len(cols_cat_and_num) == 2:
            tab_1, tab_2, tab_3, tab_4, tab_5, tab_6 = st.tabs(
                [
                    "Overview",
                    "Univariate Analysis",
                    "Missings",
                    "Anomaly Detection",
                    "Associations",
                    "Bivariate Analysis",
                ]
            )
        else:
            tab_1, tab_2, tab_3, tab_4, tab_5, tab_6, tab_7 = st.tabs(
                [
                    "Overview",
                    "Univariate Analysis",
                    "Missings",
                    "Anomaly Detection",
                    "Associations",
                    "Bivariate Analysis",
                    "Multivariate Analysis",
                ]
            )

        # Tab 1: 'Overview'
        with tab_1:
            # Create two column
            col_1, col_2 = st.columns([1, 2])
            with col_1:
                st.subheader("General Overview")
                result_eda_overview = eda_overview(data)
                st.dataframe(
                    result_eda_overview,
                    height=((len(result_eda_overview) + 1) * 35 + 3),
                )

            with col_2:
                st.subheader("Detailed Overview")
                result_eda_overview_detailed = eda_overview_detailed(
                    data, associations_df
                )
                st.dataframe(
                    result_eda_overview_detailed,
                    height=((len(result_eda_overview_detailed) + 1) * 35 + 3),
                    use_container_width=True,
                )

        # Tab 2: 'Univariate Analysis'
        with tab_2:
            selectbox_variable = st.selectbox(
                label="Select a variable", options=cols_cat_and_num, index=0
            )
            # Compute distinct values and filter alerts of the variable
            distinct_values = len(data[selectbox_variable].value_counts())
            alerts = result_eda_overview_detailed[
                result_eda_overview_detailed["Variable"] == selectbox_variable
            ][["Alert", "Detail"]].reset_index(drop=True)
            # Create two columns
            col_1, col_2 = st.columns([1, 2])
            with col_1:
                if selectbox_variable in cols_cat:
                    st.markdown(
                        "**Categorical:** " + str(distinct_values) + " distinct values"
                    )
                elif selectbox_variable in cols_num:
                    st.markdown(
                        "**Numerical:** " + str(distinct_values) + " distinct values"
                    )
                # Create two sub-columns
                col_1_1, col_1_2 = st.columns([1, 1])
                with col_1_1:
                    st.markdown("**Statistics**")
                    descriptive_statistics = data[selectbox_variable].describe()
                    st.dataframe(
                        descriptive_statistics,
                        height=((len(descriptive_statistics) + 1) * 35 + 3),
                        use_container_width=False,
                    )
                with col_1_2:
                    st.markdown("**Plotting Options**")
                    selectbox_variable_color = st.selectbox(
                        label="Select a color scale",
                        options=AVAILABLE_COLORS_SEQUENTIAL,
                        index=0,
                        key="tab_1_1",
                    )
                    selectbox_variable_template = st.selectbox(
                        label="Select a template",
                        options=AVAILABLE_TEMPLATES,
                        index=0,
                        key="tab_1_2",
                    )
                if len(alerts) > 0:
                    st.markdown("**Alerts**")
                    st.dataframe(
                        alerts,
                        height=((len(alerts) + 1) * 35 + 3),
                        use_container_width=True,
                    )
            with col_2:
                # If selected variable is CATEGORICAL
                if selectbox_variable in cols_cat:
                    # Create a DataFrame with value counts
                    data_to_be_plotted = pd.DataFrame(
                        data[selectbox_variable].value_counts()
                    ).reset_index()
                    data_to_be_plotted.columns = [selectbox_variable, "Count"]

                    if distinct_values > 3:
                        tab_2_1, tab_2_2, tab_2_3, tab_2_4, tab_2_5, tab_2_6 = st.tabs(
                            [
                                "Bar Plot",
                                "Donut Plot",
                                "Pie Plot",
                                "Polar Plot",
                                "Radar Plot",
                                "Treemap",
                            ]
                        )
                    else:
                        tab_2_1, tab_2_2, tab_2_3 = st.tabs(
                            ["Bar Plot", "Donut Plot", "Pie Plot"]
                        )
                    with tab_2_1:
                        # Create plot
                        fig_variable = plot_cat(
                            data=data_to_be_plotted,
                            var_cat=selectbox_variable,
                            var_num="Count",
                            plot_type="Bar",
                            color=selectbox_variable_color,
                            template=selectbox_variable_template,
                        )
                        st.plotly_chart(
                            fig_variable, theme="streamlit", use_container_width=True
                        )
                    with tab_2_2:
                        # Create plot
                        fig_variable = plot_cat(
                            data=data_to_be_plotted,
                            var_cat=selectbox_variable,
                            var_num="Count",
                            plot_type="Donut",
                            color=selectbox_variable_color,
                            template=selectbox_variable_template,
                        )
                        st.plotly_chart(
                            fig_variable, theme="streamlit", use_container_width=True
                        )
                    with tab_2_3:
                        # Create plot
                        fig_variable = plot_cat(
                            data=data_to_be_plotted,
                            var_cat=selectbox_variable,
                            var_num="Count",
                            plot_type="Pie",
                            color=selectbox_variable_color,
                            template=selectbox_variable_template,
                        )
                        st.plotly_chart(
                            fig_variable, theme="streamlit", use_container_width=True
                        )
                    if distinct_values > 3:
                        with tab_2_4:
                            # Create plot
                            fig_variable = plot_cat(
                                data=data_to_be_plotted,
                                var_cat=selectbox_variable,
                                var_num="Count",
                                plot_type="Polar",
                                color=selectbox_variable_color,
                                template=selectbox_variable_template,
                            )
                            st.plotly_chart(
                                fig_variable,
                                theme="streamlit",
                                use_container_width=True,
                            )
                        with tab_2_5:
                            # Create plot
                            fig_variable = plot_cat(
                                data=data_to_be_plotted,
                                var_cat=selectbox_variable,
                                var_num="Count",
                                plot_type="Radar",
                                color=selectbox_variable_color,
                                template=selectbox_variable_template,
                            )
                            st.plotly_chart(
                                fig_variable,
                                theme="streamlit",
                                use_container_width=True,
                            )
                        with tab_2_6:
                            # Create plot
                            fig_variable = plot_cat(
                                data=data_to_be_plotted,
                                var_cat=selectbox_variable,
                                var_num="Count",
                                plot_type="Treemap",
                                color=selectbox_variable_color,
                                template=selectbox_variable_template,
                            )
                            st.plotly_chart(
                                fig_variable,
                                theme="streamlit",
                                use_container_width=True,
                            )
                # If selected variable is NUMERICAL
                elif selectbox_variable in cols_num:
                    if distinct_values < 5:
                        (
                            tab_2_1,
                            tab_2_2,
                            tab_2_3,
                            tab_2_4,
                            tab_2_5,
                            tab_2_6,
                            tab_2_7,
                            tab_2_8,
                        ) = st.tabs(
                            [
                                "Histogram",
                                "KDE",
                                "Histogram + KDE",
                                "QQ-Plot",
                                "Box-Plot",
                                "Bar",
                                "Donut",
                                "Pie",
                            ]
                        )
                    else:
                        tab_2_1, tab_2_2, tab_2_3, tab_2_4, tab_2_5 = st.tabs(
                            [
                                "Histogram",
                                "KDE",
                                "Histogram + KDE",
                                "QQ-Plot",
                                "Box-Plot",
                            ]
                        )
                    with tab_2_1:
                        # Create plot
                        fig_variable = plot_num(
                            data=data,
                            var_num=selectbox_variable,
                            var_cat=None,
                            plot_type="Histogram",
                            color=selectbox_variable_color,
                            template=selectbox_variable_template,
                        )
                        st.plotly_chart(
                            fig_variable, theme="streamlit", use_container_width=True
                        )
                    with tab_2_2:
                        # Create plot
                        fig_variable = plot_num(
                            data=data,
                            var_num=selectbox_variable,
                            var_cat=None,
                            plot_type="KDE",
                            color=selectbox_variable_color,
                            template=selectbox_variable_template,
                        )
                        st.plotly_chart(
                            fig_variable, theme="streamlit", use_container_width=True
                        )
                    with tab_2_3:
                        # Create plot
                        fig_variable = plot_num(
                            data=data,
                            var_num=selectbox_variable,
                            var_cat=None,
                            plot_type="Histogram_KDE",
                            color=selectbox_variable_color,
                            template=selectbox_variable_template,
                        )
                        st.plotly_chart(
                            fig_variable, theme="streamlit", use_container_width=True
                        )
                    with tab_2_4:
                        # Create plot
                        fig_variable = plot_num(
                            data=data,
                            var_num=selectbox_variable,
                            var_cat=None,
                            plot_type="QQ-Plot",
                            color=selectbox_variable_color,
                            template=selectbox_variable_template,
                        )
                        st.plotly_chart(
                            fig_variable, theme="streamlit", use_container_width=True
                        )
                    with tab_2_5:
                        # Create plot
                        fig_variable = plot_num(
                            data=data,
                            var_num=selectbox_variable,
                            var_cat=None,
                            plot_type="Box-Plot",
                            color=selectbox_variable_color,
                            template=selectbox_variable_template,
                        )
                        st.plotly_chart(
                            fig_variable, theme="streamlit", use_container_width=True
                        )
                    if distinct_values < 5:
                        # Variable might be considered CATEGORICAL
                        data_to_be_plotted = pd.DataFrame(
                            data[selectbox_variable].value_counts()
                        ).reset_index()
                        data_to_be_plotted.columns = [selectbox_variable, "Count"]
                        with tab_2_6:
                            # Create plot
                            fig_variable = plot_cat(
                                data=data_to_be_plotted,
                                var_cat=selectbox_variable,
                                var_num="Count",
                                plot_type="Bar",
                                color=selectbox_variable_color,
                                template=selectbox_variable_template,
                            )
                            st.plotly_chart(
                                fig_variable,
                                theme="streamlit",
                                use_container_width=True,
                            )
                        with tab_2_7:
                            # Create plot
                            fig_variable = plot_cat(
                                data=data_to_be_plotted,
                                var_cat=selectbox_variable,
                                var_num="Count",
                                plot_type="Donut",
                                color=selectbox_variable_color,
                                template=selectbox_variable_template,
                            )
                            st.plotly_chart(
                                fig_variable,
                                theme="streamlit",
                                use_container_width=True,
                            )
                        with tab_2_8:
                            # Create plot
                            fig_variable = plot_cat(
                                data=data_to_be_plotted,
                                var_cat=selectbox_variable,
                                var_num="Count",
                                plot_type="Pie",
                                color=selectbox_variable_color,
                                template=selectbox_variable_template,
                            )
                            st.plotly_chart(
                                fig_variable,
                                theme="streamlit",
                                use_container_width=True,
                            )

        # Tab 3: 'Missings'
        with tab_3:
            # Compute DataFrame to show the Missing Overview and a list with variables that contain missing values
            missing_overview = result_eda_overview[
                result_eda_overview.index == "Number of missing cells"
            ]
            missing_overview = pd.concat(
                [
                    missing_overview,
                    result_eda_overview[
                        result_eda_overview.index == "Missing cells (%)"
                    ],
                ],
                axis=0,
            )
            vars_with_missings = result_eda_overview_detailed[
                result_eda_overview_detailed["Alert"] == "Missing Values"
            ]["Variable"].to_list()
            if len(vars_with_missings) == 0:
                # Show results and selectboxes
                st.markdown("**General Overview**")
                st.dataframe(
                    missing_overview,
                    height=((len(missing_overview) + 1) * 35 + 3),
                    use_container_width=False,
                )
            else:
                # Create several tabs:
                tab_3_1, tab_3_2, tab_3_3 = st.tabs(
                    [
                        "Overview",
                        "Associations + Data including encoded missings",
                        "Bivariate Analysis",
                    ]
                )
                # Tab 6_1: Overview
                with tab_3_1:
                    # Create two column
                    col_1, col_2 = st.columns([1, 2])
                    with col_1:
                        list_of_numbers_of_missings = []
                        # Loop through the list and create a DataFrame to be plotted
                        for column in vars_with_missings:
                            list_of_numbers_of_missings.append(
                                np.array(data[column].isna()).sum()
                            )
                        missing_details = pd.DataFrame()
                        missing_details["Variable"] = vars_with_missings
                        missing_details["Missings"] = list_of_numbers_of_missings
                        # Show results and selectboxes
                        st.markdown("**General Overview**")
                        st.dataframe(
                            missing_overview,
                            height=((len(missing_overview) + 1) * 35 + 3),
                            use_container_width=False,
                        )
                        st.markdown("**Plotting Options**")
                        selectbox_variable_color = st.selectbox(
                            label="Select a color scale",
                            options=AVAILABLE_COLORS_SEQUENTIAL,
                            index=0,
                            key="tab_6_1_1",
                        )
                        selectbox_variable_template = st.selectbox(
                            label="Select a template",
                            options=AVAILABLE_TEMPLATES,
                            index=0,
                            key="tab_6_1_2",
                        )
                    with col_2:
                        fig_variable = plot_cat(
                            data=missing_details.sort_values(
                                by="Missings", ascending=False
                            ),
                            var_cat="Variable",
                            var_num="Missings",
                            plot_type="Bar",
                            color=selectbox_variable_color,
                            template=selectbox_variable_template,
                        )
                        st.plotly_chart(
                            fig_variable, theme="streamlit", use_container_width=True
                        )
                # Tab 3_2: 'Associations' + 'DataFrame including encoded missings'
                with tab_3_2:
                    selectbox_associations_color = st.selectbox(
                        label="Select a color scale",
                        options=list(AVAILABLE_COLORS_DIVERGING.keys()),
                        index=0,
                        key="tab_6_2_1",
                    )
                    # Create a new DataFrame that includes new columns for present - missing
                    data_for_missing = data.copy(deep=True)
                    vars_with_recoded_missings = []
                    for column in vars_with_missings:
                        string_new_name = "Missing_in_" + str(column)
                        vars_with_recoded_missings.append(string_new_name)
                        data_for_missing.loc[
                            data_for_missing[column] != data_for_missing[column],
                            string_new_name,
                        ] = "missing"
                        data_for_missing.loc[
                            data_for_missing[column] == data_for_missing[column],
                            string_new_name,
                        ] = "present"
                    # Get CATEGORICAL column names
                    cols_cat_missing = data_for_missing.select_dtypes(
                        include=["object", "category", "bool"]
                    ).columns.to_list()
                    # Compute associations (necesary for the overview tab)
                    associations_missings = (
                        associations_for_categorical_and_numerical_variables(
                            data_for_missing
                        )
                    )[0]
                    mask = np.triu(np.ones_like(associations_missings, dtype=bool))
                    associations_missings = associations_missings.mask(mask)
                    associations_missings = associations_missings.iloc[
                        -len(vars_with_recoded_missings) :  # noqa: E203
                    ]
                    # Plot the Heatmap
                    fig_correlation = plot_heatmap(
                        associations=associations_missings,
                        color=selectbox_associations_color,
                        zmin=-1,
                        zmax=1,
                    )
                    st.markdown("**Associations**")
                    st.plotly_chart(
                        fig_correlation, theme="streamlit", use_container_width=True
                    )
                    st.markdown("**DataFrame including encoded missings**")
                    selectbox_variables = st.multiselect(
                        label="Select the variables",
                        options=data_for_missing.columns.to_list(),
                    )
                    if len(selectbox_variables) > 0:
                        st.dataframe(
                            data_for_missing[selectbox_variables],
                            use_container_width=False,
                        )
                # Tab 6_3: Bivariate Analysis
                with tab_3_3:
                    # Create two columns
                    col_3_3_1, col_3_3_2 = st.columns([1, 1])
                    with col_3_3_1:
                        selectbox_target = st.selectbox(
                            label="Select the target variable",
                            options=vars_with_recoded_missings,
                            index=0,
                        )
                    with col_3_3_2:
                        string_selectbox_target_new_name = (
                            selectbox_target.removeprefix("Missing_in_")
                        )
                        available_features = [
                            value
                            for value in data_for_missing.columns.to_list()
                            if value
                            not in [selectbox_target, string_selectbox_target_new_name]
                        ]
                        selectbox_feature = st.selectbox(
                            label="Select a feature",
                            options=available_features,
                            index=0,
                        )
                    # Create two columns
                    col_3_3_1, col_3_3_2 = st.columns([1, 4])
                    with col_3_3_1:
                        st.markdown("**Plotting Options**")
                        selectbox_variable_color = st.selectbox(
                            label="**Select a color scale**",
                            options=AVAILABLE_COLORS_SEQUENTIAL,
                            index=0,
                            key="col_3_3_1_c",
                        )
                        selectbox_variable_template = st.selectbox(
                            label="**Select a template**",
                            options=AVAILABLE_TEMPLATES,
                            index=0,
                            key="col_3_3_1_t",
                        )
                    # Plots
                    with col_3_3_2:
                        # CAT & CAT
                        if (selectbox_target in cols_cat_missing) & (
                            selectbox_feature in cols_cat_missing
                        ):
                            # Create tabs for plots
                            tab_3_3_1, tab_3_3_2, tab_3_3_3 = st.tabs(
                                [
                                    "Grouped Bar Plot",
                                    "Stacked Bar Plot",
                                    "100% Stacked Bar Plot",
                                ]
                            )
                            with tab_3_3_1:
                                # Create plot
                                fig_variable = plot_cat_cat(
                                    data=data_for_missing,
                                    target=selectbox_target,
                                    feature=selectbox_feature,
                                    barmode="group",
                                    color=selectbox_variable_color,
                                    template=selectbox_variable_template,
                                )
                                st.plotly_chart(
                                    fig_variable,
                                    theme="streamlit",
                                    use_container_width=True,
                                )
                            with tab_3_3_2:
                                # Create plot
                                fig_variable = plot_cat_cat(
                                    data=data_for_missing,
                                    target=selectbox_target,
                                    feature=selectbox_feature,
                                    barmode="stack",
                                    color=selectbox_variable_color,
                                    template=selectbox_variable_template,
                                )
                                st.plotly_chart(
                                    fig_variable,
                                    theme="streamlit",
                                    use_container_width=True,
                                )
                            with tab_3_3_3:
                                # Create plot
                                fig_variable = plot_cat_cat(
                                    data=data_for_missing,
                                    target=selectbox_target,
                                    feature=selectbox_feature,
                                    barmode="100_stack",
                                    color=selectbox_variable_color,
                                    template=selectbox_variable_template,
                                )
                                st.plotly_chart(
                                    fig_variable,
                                    theme="streamlit",
                                    use_container_width=True,
                                )
                        # CAT & NUM
                        else:
                            # Create CATEGORICAL and NUMERICAL variable for plots
                            if selectbox_target in cols_cat_missing:
                                var_cat = selectbox_target
                                var_num = selectbox_feature
                            else:
                                var_cat = selectbox_feature
                                var_num = selectbox_target
                            # Create a Dataframe for plots based on CATEGORICAL variables and compute distinct values
                            data_for_cat_plots = (
                                pd.DataFrame(data_for_missing[[var_cat, var_num]])
                                .groupby(by=var_cat)
                                .mean()
                                .reset_index()
                            )
                            distinct_values = len(
                                data_for_missing[var_cat].value_counts()
                            )
                            # Create tabs for plots
                            if distinct_values > 3:
                                (
                                    tab_3_3_1,
                                    tab_3_3_2,
                                    tab_3_3_3,
                                    tab_3_3_4,
                                    tab_3_3_5,
                                    tab_3_3_6,
                                    tab_3_3_7,
                                    tab_3_3_8,
                                ) = st.tabs(
                                    [
                                        "Bar Plot",
                                        "Box-Plot",
                                        "Histogram",
                                        "KDE",
                                        "Histogram + KDE",
                                        "Polar Plot",
                                        "Radar Plot",
                                        "Treemap",
                                    ]
                                )
                            else:
                                (
                                    tab_3_3_1,
                                    tab_3_3_2,
                                    tab_3_3_3,
                                    tab_3_3_4,
                                    tab_3_3_5,
                                ) = st.tabs(
                                    [
                                        "Bar Plot",
                                        "Box-Plot",
                                        "Histogram",
                                        "KDE",
                                        "Histogram + KDE",
                                    ]
                                )
                            with tab_3_3_1:
                                fig_variable = plot_cat(
                                    data=data_for_cat_plots,
                                    var_cat=var_cat,
                                    var_num=var_num,
                                    plot_type="Bar",
                                    color=selectbox_variable_color,
                                    template=selectbox_variable_template,
                                )
                                st.plotly_chart(
                                    fig_variable,
                                    theme="streamlit",
                                    use_container_width=True,
                                )
                            with tab_3_3_2:
                                fig_variable = plot_num(
                                    data=data_for_missing,
                                    var_num=var_num,
                                    var_cat=var_cat,
                                    plot_type="Box-Plot",
                                    color=selectbox_variable_color,
                                    template=selectbox_variable_template,
                                )
                                st.plotly_chart(
                                    fig_variable,
                                    theme="streamlit",
                                    use_container_width=True,
                                )
                            with tab_3_3_3:
                                fig_variable = plot_num(
                                    data=data_for_missing,
                                    var_num=var_num,
                                    var_cat=var_cat,
                                    plot_type="Histogram",
                                    color=selectbox_variable_color,
                                    template=selectbox_variable_template,
                                )
                                st.plotly_chart(
                                    fig_variable,
                                    theme="streamlit",
                                    use_container_width=True,
                                )
                            with tab_3_3_4:
                                fig_variable = plot_num(
                                    data=data_for_missing,
                                    var_num=var_num,
                                    var_cat=var_cat,
                                    plot_type="KDE",
                                    color=selectbox_variable_color,
                                    template=selectbox_variable_template,
                                )
                                st.plotly_chart(
                                    fig_variable,
                                    theme="streamlit",
                                    use_container_width=True,
                                )
                            with tab_3_3_5:
                                fig_variable = plot_num(
                                    data=data_for_missing,
                                    var_num=var_num,
                                    var_cat=var_cat,
                                    plot_type="Histogram_KDE",
                                    color=selectbox_variable_color,
                                    template=selectbox_variable_template,
                                )
                                st.plotly_chart(
                                    fig_variable,
                                    theme="streamlit",
                                    use_container_width=True,
                                )
                            if distinct_values > 3:
                                with tab_3_3_6:
                                    fig_variable = plot_cat(
                                        data=data_for_cat_plots,
                                        var_cat=var_cat,
                                        var_num=var_num,
                                        plot_type="Polar",
                                        color=selectbox_variable_color,
                                        template=selectbox_variable_template,
                                    )
                                    st.plotly_chart(
                                        fig_variable,
                                        theme="streamlit",
                                        use_container_width=True,
                                    )
                                with tab_3_3_7:
                                    fig_variable = plot_cat(
                                        data=data_for_cat_plots,
                                        var_cat=var_cat,
                                        var_num=var_num,
                                        plot_type="Radar",
                                        color=selectbox_variable_color,
                                        template=selectbox_variable_template,
                                    )
                                    st.plotly_chart(
                                        fig_variable,
                                        theme="streamlit",
                                        use_container_width=True,
                                    )
                                with tab_3_3_8:
                                    fig_variable = plot_cat(
                                        data=data_for_cat_plots,
                                        var_cat=var_cat,
                                        var_num=var_num,
                                        plot_type="Treemap",
                                        color=selectbox_variable_color,
                                        template=selectbox_variable_template,
                                    )
                                    st.plotly_chart(
                                        fig_variable,
                                        theme="streamlit",
                                        use_container_width=True,
                                    )
        # Tab 4: 'Anomaly Detection'
        if 1 < len(cols_cat_and_num):
            with tab_4:
                # Create two columns: to select a model and to display it)
                col_ad_1, col_ad_2 = st.columns([1, 3])

        # Tab 4: 'Associations'
        if 1 < len(cols_cat_and_num):
            with tab_4:
                st.warning(
                    "Calculate the strength-of-association of features in data-set with both, categorical"
                    " and continuous features using: Spearman's R for continuous-continuous cases - Correlation Ratio"
                    " for categorical-continuous cases - bias corrected Cramer's V for categorical-categorical cases."
                    " For more information of the bias corrected Cramer's V, see:  \nBergsma, W. (2013)."
                    " A bias-correction for Cramér's and Tschuprow's. Journal of the Korean Statistical Society, 42(3),"
                    " 323-328. https://doi.org/10.1016/j.jkss.2012.10.002"
                )
                tab_4_1, tab_4_2 = st.tabs(["Heatmap", "Association matrix"])
                # Heatmap
                with tab_4_1:
                    # Create two columns
                    selectbox_associations_color = st.selectbox(
                        label="Select a color scale",
                        options=list(AVAILABLE_COLORS_DIVERGING.keys()),
                        index=0,
                        key="tab_4",
                    )
                    # Create a mask for the association matrix and plot the Heatmap
                    mask = np.triu(np.ones_like(associations_df, dtype=bool))
                    associations_to_be_plotted = associations_df.mask(mask)
                    fig_correlation = plot_heatmap(
                        associations=associations_to_be_plotted,
                        color=selectbox_associations_color,
                        zmin=-1,
                        zmax=1,
                    )
                    st.plotly_chart(
                        fig_correlation, theme="streamlit", use_container_width=True
                    )
                # Association matrix
                with tab_4_2:
                    # Create two columns
                    col_4_2_1, col_4_2_2 = st.columns([1, 3])
                    with col_4_2_1:
                        st.download_button(
                            label="Download associations as CSV",
                            data=convert_dataframe_to_csv(associations_df),
                            file_name="associations.csv",
                            mime="text/csv'",
                        )
                    with col_4_2_2:
                        st.download_button(
                            label="Download associations as XLSX",
                            data=convert_dataframe_to_xlsx(associations_df),
                            file_name="associations.xlsx",
                            mime="application/vnd.ms-excel",
                        )

                    st.dataframe(
                        associations_df, height=((len(associations_df) + 1) * 35 + 3)
                    )

        # Tab 5: 'Bivariate Analysis'
        if 1 < len(cols_cat_and_num):
            with tab_5:
                # Create two columns
                col_5_1, col_5_2 = st.columns([1, 1])
                with col_5_1:
                    selectbox_target = st.selectbox(
                        label="Select the target variable",
                        options=cols_cat_and_num,
                        index=0,
                    )
                with col_5_2:
                    available_features = [
                        value for value in cols_cat_and_num if value != selectbox_target
                    ]
                    selectbox_feature = st.selectbox(
                        label="Select a feature", options=available_features, index=0
                    )
                # Create two columns
                col_5_1, col_5_2 = st.columns([1, 4])
                with col_5_1:
                    st.markdown("**Plotting Options**")
                    selectbox_variable_color = st.selectbox(
                        label="**Select a color scale**",
                        options=AVAILABLE_COLORS_SEQUENTIAL,
                        index=0,
                    )
                    selectbox_variable_template = st.selectbox(
                        label="**Select a template**",
                        options=AVAILABLE_TEMPLATES,
                        index=0,
                    )
                    if (selectbox_target in cols_num) & (selectbox_feature in cols_num):
                        selectbox_marginal_x = st.selectbox(
                            label="**Select a Marginal Distribution Plot for the feature**",
                            options=["histogram", "rug", "box", "violin", None],
                            index=0,
                        )
                        selectbox_marginal_y = st.selectbox(
                            label="**Select a Marginal Distribution Plot for the target variable**",
                            options=["histogram", "rug", "box", "violin", None],
                            index=0,
                        )
                # Plots
                with col_5_2:
                    # CAT & CAT
                    if (selectbox_target in cols_cat) & (selectbox_feature in cols_cat):
                        # Create tabs for plots
                        tab_5_1, tab_5_2, tab_5_3 = st.tabs(
                            [
                                "Grouped Bar Plot",
                                "Stacked Bar Plot",
                                "100% Stacked Bar Plot",
                            ]
                        )
                        with tab_5_1:
                            # Create plot
                            fig_variable = plot_cat_cat(
                                data=data,
                                target=selectbox_target,
                                feature=selectbox_feature,
                                barmode="group",
                                color=selectbox_variable_color,
                                template=selectbox_variable_template,
                            )
                            st.plotly_chart(
                                fig_variable,
                                theme="streamlit",
                                use_container_width=True,
                            )
                        with tab_5_2:
                            # Create plot
                            fig_variable = plot_cat_cat(
                                data=data,
                                target=selectbox_target,
                                feature=selectbox_feature,
                                barmode="stack",
                                color=selectbox_variable_color,
                                template=selectbox_variable_template,
                            )
                            st.plotly_chart(
                                fig_variable,
                                theme="streamlit",
                                use_container_width=True,
                            )
                        with tab_5_3:
                            # Create plot
                            fig_variable = plot_cat_cat(
                                data=data,
                                target=selectbox_target,
                                feature=selectbox_feature,
                                barmode="100_stack",
                                color=selectbox_variable_color,
                                template=selectbox_variable_template,
                            )
                            st.plotly_chart(
                                fig_variable,
                                theme="streamlit",
                                use_container_width=True,
                            )
                    # NUM & NUM
                    elif (selectbox_target in cols_num) & (
                        selectbox_feature in cols_num
                    ):
                        # Create tabs for plots
                        tab_5_1, tab_5_2, tab_5_3 = st.tabs(
                            ["OLS trendline", "LOWESS trendline", "None"]
                        )
                        with tab_5_1:
                            fig_variable = plot_num_num(
                                data=data,
                                target=selectbox_target,
                                feature=selectbox_feature,
                                trendline="ols",
                                plot_type="scatter",
                                marginal_x=selectbox_marginal_x,
                                marginal_y=selectbox_marginal_y,
                                color=selectbox_variable_color,
                                template=selectbox_variable_template,
                            )
                            st.plotly_chart(
                                fig_variable,
                                theme="streamlit",
                                use_container_width=True,
                            )
                        with tab_5_2:
                            fig_variable = plot_num_num(
                                data=data,
                                target=selectbox_target,
                                feature=selectbox_feature,
                                trendline="lowess",
                                plot_type="scatter",
                                marginal_x=selectbox_marginal_x,
                                marginal_y=selectbox_marginal_y,
                                color=selectbox_variable_color,
                                template=selectbox_variable_template,
                            )
                            st.plotly_chart(
                                fig_variable,
                                theme="streamlit",
                                use_container_width=True,
                            )
                        with tab_5_3:
                            fig_variable = plot_num_num(
                                data=data,
                                target=selectbox_target,
                                feature=selectbox_feature,
                                trendline=None,
                                plot_type="scatter",
                                marginal_x=selectbox_marginal_x,
                                marginal_y=selectbox_marginal_y,
                                color=selectbox_variable_color,
                                template=selectbox_variable_template,
                            )
                            st.plotly_chart(
                                fig_variable,
                                theme="streamlit",
                                use_container_width=True,
                            )

                    # CAT & NUM
                    else:
                        # Create CATEGORICAL and NUMERICAL variable for plots
                        if selectbox_target in cols_cat:
                            var_cat = selectbox_target
                            var_num = selectbox_feature
                        else:
                            var_cat = selectbox_feature
                            var_num = selectbox_target
                        # Create a Dataframe for plots based on CATEGORICAL variables and compute distinct values
                        data_for_cat_plots = (
                            pd.DataFrame(data[[var_cat, var_num]])
                            .groupby(by=var_cat)
                            .mean()
                            .reset_index()
                        )
                        distinct_values = len(data[var_cat].value_counts())
                        # Create tabs for plots
                        if distinct_values > 3:
                            (
                                tab_5_1,
                                tab_5_2,
                                tab_5_3,
                                tab_5_4,
                                tab_5_5,
                                tab_5_6,
                                tab_5_7,
                                tab_5_8,
                            ) = st.tabs(
                                [
                                    "Bar Plot",
                                    "Box-Plot",
                                    "Histogram",
                                    "KDE",
                                    "Histogram + KDE",
                                    "Polar Plot",
                                    "Radar Plot",
                                    "Treemap",
                                ]
                            )
                        else:
                            tab_5_1, tab_5_2, tab_5_3, tab_5_4, tab_5_5 = st.tabs(
                                [
                                    "Bar Plot",
                                    "Box-Plot",
                                    "Histogram",
                                    "KDE",
                                    "Histogram + KDE",
                                ]
                            )
                        with tab_5_1:
                            fig_variable = plot_cat(
                                data=data_for_cat_plots,
                                var_cat=var_cat,
                                var_num=var_num,
                                plot_type="Bar",
                                color=selectbox_variable_color,
                                template=selectbox_variable_template,
                            )
                            st.plotly_chart(
                                fig_variable,
                                theme="streamlit",
                                use_container_width=True,
                            )
                        with tab_5_2:
                            fig_variable = plot_num(
                                data=data,
                                var_num=var_num,
                                var_cat=var_cat,
                                plot_type="Box-Plot",
                                color=selectbox_variable_color,
                                template=selectbox_variable_template,
                            )
                            st.plotly_chart(
                                fig_variable,
                                theme="streamlit",
                                use_container_width=True,
                            )
                        with tab_5_3:
                            fig_variable = plot_num(
                                data=data,
                                var_num=var_num,
                                var_cat=var_cat,
                                plot_type="Histogram",
                                color=selectbox_variable_color,
                                template=selectbox_variable_template,
                            )
                            st.plotly_chart(
                                fig_variable,
                                theme="streamlit",
                                use_container_width=True,
                            )
                        with tab_5_4:
                            fig_variable = plot_num(
                                data=data,
                                var_num=var_num,
                                var_cat=var_cat,
                                plot_type="KDE",
                                color=selectbox_variable_color,
                                template=selectbox_variable_template,
                            )
                            st.plotly_chart(
                                fig_variable,
                                theme="streamlit",
                                use_container_width=True,
                            )
                        with tab_5_5:
                            fig_variable = plot_num(
                                data=data,
                                var_num=var_num,
                                var_cat=var_cat,
                                plot_type="Histogram_KDE",
                                color=selectbox_variable_color,
                                template=selectbox_variable_template,
                            )
                            st.plotly_chart(
                                fig_variable,
                                theme="streamlit",
                                use_container_width=True,
                            )
                        if distinct_values > 3:
                            with tab_5_6:
                                fig_variable = plot_cat(
                                    data=data_for_cat_plots,
                                    var_cat=var_cat,
                                    var_num=var_num,
                                    plot_type="Polar",
                                    color=selectbox_variable_color,
                                    template=selectbox_variable_template,
                                )
                                st.plotly_chart(
                                    fig_variable,
                                    theme="streamlit",
                                    use_container_width=True,
                                )
                            with tab_5_7:
                                fig_variable = plot_cat(
                                    data=data_for_cat_plots,
                                    var_cat=var_cat,
                                    var_num=var_num,
                                    plot_type="Radar",
                                    color=selectbox_variable_color,
                                    template=selectbox_variable_template,
                                )
                                st.plotly_chart(
                                    fig_variable,
                                    theme="streamlit",
                                    use_container_width=True,
                                )
                            with tab_5_8:
                                fig_variable = plot_cat(
                                    data=data_for_cat_plots,
                                    var_cat=var_cat,
                                    var_num=var_num,
                                    plot_type="Treemap",
                                    color=selectbox_variable_color,
                                    template=selectbox_variable_template,
                                )
                                st.plotly_chart(
                                    fig_variable,
                                    theme="streamlit",
                                    use_container_width=True,
                                )
        # Tab 6: 'Multivariate Analysis'
        if 2 < len(cols_num):
            with tab_6:
                # Create several tabs:
                tab_6_1, tab_6_2, tab_6_3, tab_6_4, tab_6_5 = st.tabs(
                    [
                        "Mean Based Plots",
                        "Bubble Chart",
                        "Random Feature Dropping",
                        "PCA Projection",
                        "Manifold Visualization",
                    ]
                )
                # Tab 6_1: Mean Based Plots
                with tab_6_1:
                    # Create two columns
                    col_6_1, col_6_2 = st.columns([1, 1])
                    with col_6_1:
                        selectbox_variables = st.multiselect(
                            label="Select the variables",
                            options=cols_num,
                            default=[cols_num[0], cols_num[1]],
                        )
                    with col_6_2:
                        selectbox_grouping_variable = st.selectbox(
                            label="Select a grouping variable",
                            options=[None] + cols_cat,
                            index=1,
                        )
                    # Create two columns
                    col_6_1, col_6_2 = st.columns([1, 4])
                    with col_6_1:
                        st.markdown("**Plotting Options**")
                        selectbox_variable_color = st.selectbox(
                            label="**Select a color scale**",
                            options=AVAILABLE_COLORS_SEQUENTIAL,
                            index=0,
                            key="tab_6_1_color",
                        )
                        selectbox_variable_template = st.selectbox(
                            label="**Select a template**",
                            options=AVAILABLE_TEMPLATES,
                            index=0,
                            key="tab_6_1_template",
                        )
                    # Plots
                    with col_6_2:
                        # Without grouping variable
                        if selectbox_grouping_variable is None:
                            # Create a Dataframe for plots based on NUMERICAL variables
                            data_to_be_plotted = pd.DataFrame(
                                data[selectbox_variables].mean()
                            ).reset_index()
                            data_to_be_plotted.columns = ["Variables", "Mean"]
                            # Create tabs for plots
                            tab_6_1_1, tab_6_1_2, tab_6_1_3 = st.tabs(
                                ["Bar Plot", "Polar Plot", "Treemap"]
                            )
                            with tab_6_1_1:
                                fig_variable = plot_cat(
                                    data=data_to_be_plotted.sort_values(
                                        by="Mean", ascending=False
                                    ),
                                    var_cat="Variables",
                                    var_num="Mean",
                                    plot_type="Bar",
                                    color=selectbox_variable_color,
                                    template=selectbox_variable_template,
                                )
                                st.plotly_chart(
                                    fig_variable,
                                    theme="streamlit",
                                    use_container_width=True,
                                )
                            with tab_6_1_2:
                                fig_variable = plot_cat(
                                    data=data_to_be_plotted.sort_values(
                                        by="Mean", ascending=False
                                    ),
                                    var_cat="Variables",
                                    var_num="Mean",
                                    plot_type="Polar",
                                    color=selectbox_variable_color,
                                    template=selectbox_variable_template,
                                )
                                st.plotly_chart(
                                    fig_variable,
                                    theme="streamlit",
                                    use_container_width=True,
                                )
                            with tab_6_1_3:
                                fig_variable = plot_cat(
                                    data=data_to_be_plotted.sort_values(
                                        by="Mean", ascending=False
                                    ),
                                    var_cat="Variables",
                                    var_num="Mean",
                                    plot_type="Treemap",
                                    color=selectbox_variable_color,
                                    template=selectbox_variable_template,
                                )
                                st.plotly_chart(
                                    fig_variable,
                                    theme="streamlit",
                                    use_container_width=True,
                                )
                        else:
                            # Create tabs for plots
                            tab_6_1_1, tab_6_1_2, tab_6_1_3 = st.tabs(
                                [
                                    "Grouped Bar Plot",
                                    "Stacked Bar Plot",
                                    "100% Stacked Bar Plot",
                                ]
                            )
                            with tab_6_1_1:
                                # Create plot
                                fig_variable = plot_num_with_grouping_variable(
                                    data=data,
                                    var_num=selectbox_variables,
                                    var_group=selectbox_grouping_variable,
                                    barmode="group",
                                    color=selectbox_variable_color,
                                    template=selectbox_variable_template,
                                )
                                st.plotly_chart(
                                    fig_variable,
                                    theme="streamlit",
                                    use_container_width=True,
                                )
                            with tab_6_1_2:
                                # Create plot
                                fig_variable = plot_num_with_grouping_variable(
                                    data=data,
                                    var_num=selectbox_variables,
                                    var_group=selectbox_grouping_variable,
                                    barmode="stack",
                                    color=selectbox_variable_color,
                                    template=selectbox_variable_template,
                                )
                                st.plotly_chart(
                                    fig_variable,
                                    theme="streamlit",
                                    use_container_width=True,
                                )
                            with tab_6_1_3:
                                # Create plot
                                fig_variable = plot_num_with_grouping_variable(
                                    data=data,
                                    var_num=selectbox_variables,
                                    var_group=selectbox_grouping_variable,
                                    barmode="100_stack",
                                    color=selectbox_variable_color,
                                    template=selectbox_variable_template,
                                )
                                st.plotly_chart(
                                    fig_variable,
                                    theme="streamlit",
                                    use_container_width=True,
                                )
                # Tab 6_2: Bubble Chart
                with tab_6_2:
                    # Create two columns
                    col_6_1, col_6_2 = st.columns([1, 1])
                    with col_6_1:
                        selectbox_x_axis = st.selectbox(
                            label="Select the variable to be plotted on the x-axis",
                            options=cols_num,
                        )
                    with col_6_2:
                        available_for_y_axis = [
                            value for value in cols_num if value != selectbox_x_axis
                        ]
                        selectbox_y_axis = st.selectbox(
                            label="Select the variable to be plotted on the y-axis",
                            options=available_for_y_axis,
                            index=0,
                        )
                    # Create two columns
                    col_6_1, col_6_2 = st.columns([1, 4])
                    with col_6_1:
                        available_for_size = [
                            value
                            for value in available_for_y_axis
                            if value != selectbox_y_axis
                        ]
                        selectbox_size = st.selectbox(
                            label="Select the variable that determines the size of the bubbles",
                            options=[None] + available_for_size,
                            index=0,
                        )

                        selectbox_color = st.selectbox(
                            label="Select the variable that determines the color of the bubbles",
                            options=[None] + cols_cat,
                            index=0,
                        )

                        available_for_hover_name = [
                            value for value in cols_cat if value != selectbox_color
                        ]
                        selectbox_hover_name = st.selectbox(
                            label="Select the variable that is displayed in bold as the tooltip title",
                            options=[None] + available_for_hover_name,
                            index=0,
                        )

                        st.markdown("**Plotting Options**")
                        selectbox_variable_color = st.selectbox(
                            label="**Select a color scale**",
                            options=AVAILABLE_COLORS_SEQUENTIAL,
                            index=0,
                            key="tab_6_2_color",
                        )
                        selectbox_variable_template = st.selectbox(
                            label="**Select a template**",
                            options=AVAILABLE_TEMPLATES,
                            index=0,
                            key="tab_6_2_template",
                        )
                    # Plots
                    with col_6_2:
                        fig_variable = plot_bubble_chart(
                            data,
                            x=selectbox_x_axis,
                            y=selectbox_y_axis,
                            var_size=selectbox_size,
                            var_color=selectbox_color,
                            var_hover_name=selectbox_hover_name,
                            color=selectbox_variable_color,
                            template=selectbox_variable_template,
                        )
                        st.plotly_chart(
                            fig_variable, theme="streamlit", use_container_width=True
                        )
                # Tab 6_3: Random Feature Dropping
                with tab_6_3:
                    # Create two columns
                    col_6_1, col_6_2 = st.columns([1, 4])
                    with col_6_1:
                        selectbox_target = st.selectbox(
                            label="Select the target variable",
                            options=cols_cat_and_num,
                            key="tab_6_3_rfd",
                        )
                        # Initiate 'operation' and 'evaluation_score_options'
                        if selectbox_target in cols_cat:
                            operation = "classification"
                            evaluation_score_options = AVAILABLE_SCORES_CLASSIFICATION
                        else:
                            operation = "regression"
                            evaluation_score_options = AVAILABLE_SCORES_REGRESSION
                        selectbox_n_cv_folds = st.selectbox(
                            label="Select the number of cross validation folds",
                            options=range(5, 11),
                            index=5,
                        )
                        selectbox_evaluation_score = st.selectbox(
                            label="Select the evaluation score",
                            options=evaluation_score_options,
                        )
                        if selectbox_evaluation_score in [
                            "AUC",
                            "Recall",
                            "Precision",
                            "F1",
                        ]:
                            selectbox_average = st.selectbox(
                                label="Select the average to be used",
                                options=["micro", "macro", "weighted"],
                                index=1,
                            )
                        else:
                            selectbox_average = []
                        # Initiate a placeholder for the figure
                        if "fig_rfdc" not in st.session_state:
                            st.session_state.fig_rfdc = None

                        if st.button(
                            "Generate plot",
                            type="primary",
                            use_container_width=True,
                            key="button_plot_fig_rfdc",
                        ):
                            st.session_state.fig_rfdc = (
                                plot_random_feature_dropping_curve(
                                    data=data,
                                    target_variable=selectbox_target,
                                    operation=operation,
                                    evaluation_score=selectbox_evaluation_score,
                                    average=selectbox_average,
                                    cv_folds=selectbox_n_cv_folds,
                                )
                            )
                    with col_6_2:
                        if st.session_state.fig_rfdc is not None:
                            components.html(
                                st.session_state.fig_rfdc,
                                height=600,
                            )

                # Tab 6_4: PCA Proyection
                with tab_6_4:
                    # Create a PCA instance
                    selectbox_target = st.selectbox(
                        label="**Select the target variable**",
                        options=cols_cat_and_num,
                        key="tab_6_4_pca",
                    )
                    pca_instance = principal_component_analysis(
                        data=data, target_variable=selectbox_target
                    )
                    # Create tabs for plots
                    tab_6_4_1, tab_6_4_2, tab_6_4_3, tab_6_4_4, tab_6_4_5 = st.tabs(
                        [
                            "PCA plot - 2D",
                            "PCA plot - 3D",
                            "Explained Variance Plot",
                            "Loadings Matrix",
                            "Transformed Data",
                        ]
                    )
                    # PCA plot - 2D
                    with tab_6_4_1:
                        # Create two columns
                        col_6_1, col_6_2 = st.columns([1, 4])
                        with col_6_1:
                            st.markdown("**Plotting Options**")
                            selectbox_variable_color = st.selectbox(
                                label="**Select a color scale**",
                                options=AVAILABLE_COLORS_SEQUENTIAL,
                                index=0,
                                key="tab_6_4_1_color",
                            )
                            selectbox_variable_template = st.selectbox(
                                label="**Select a template**",
                                options=AVAILABLE_TEMPLATES,
                                index=0,
                                key="tab_6_4_1_template",
                            )
                        with col_6_2:
                            exp_var_pc_1 = (
                                pca_instance.data_explained_variances["explained"][0]
                                * 100
                            )
                            exp_var_pc_2 = (
                                pca_instance.data_explained_variances["explained"][1]
                                * 100
                            )
                            fig_variable = plot_pca_2d(
                                data=data,
                                data_pca=pca_instance.data_pca,
                                exp_var_pc_1=exp_var_pc_1,
                                exp_var_pc_2=exp_var_pc_2,
                                target=selectbox_target,
                                color=selectbox_variable_color,
                                template=selectbox_variable_template,
                            )
                            st.plotly_chart(
                                fig_variable,
                                theme="streamlit",
                                use_container_width=True,
                            )
                    # PCA plot - 3D
                    with tab_6_4_2:
                        # Create two columns
                        col_6_1, col_6_2 = st.columns([1, 4])
                        with col_6_1:
                            st.markdown("**Plotting Options**")
                            selectbox_variable_color = st.selectbox(
                                label="**Select a color scale**",
                                options=AVAILABLE_COLORS_SEQUENTIAL,
                                index=0,
                                key="tab_6_4_2_color",
                            )
                            selectbox_variable_template = st.selectbox(
                                label="**Select a template**",
                                options=AVAILABLE_TEMPLATES,
                                index=0,
                                key="tab_6_4_2_template",
                            )
                        with col_6_2:
                            exp_var_pc_1 = (
                                pca_instance.data_explained_variances["explained"][0]
                                * 100
                            )
                            exp_var_pc_2 = (
                                pca_instance.data_explained_variances["explained"][1]
                                * 100
                            )
                            exp_var_pc_3 = (
                                pca_instance.data_explained_variances["explained"][2]
                                * 100
                            )
                            fig_variable = plot_pca_3d(
                                data=data,
                                data_pca=pca_instance.data_pca,
                                exp_var_pc_1=exp_var_pc_1,
                                exp_var_pc_2=exp_var_pc_2,
                                exp_var_pc_3=exp_var_pc_3,
                                target=selectbox_target,
                                color=selectbox_variable_color,
                                template=selectbox_variable_template,
                            )
                            st.plotly_chart(
                                fig_variable,
                                theme="streamlit",
                                use_container_width=True,
                            )
                    # Explained Variance Plot
                    with tab_6_4_3:
                        # Create two columns
                        col_6_1, col_6_2 = st.columns([1, 4])
                        with col_6_1:
                            st.markdown("**Plotting Options**")
                            selectbox_variable_color = st.selectbox(
                                label="**Select a color scale**",
                                options=AVAILABLE_COLORS_SEQUENTIAL,
                                index=0,
                                key="tab_6_4_3_color",
                            )
                            selectbox_variable_template = st.selectbox(
                                label="**Select a template**",
                                options=AVAILABLE_TEMPLATES,
                                index=0,
                                key="tab_6_4_3_template",
                            )
                        with col_6_2:
                            # Compute a DataFrame for the 'Explained Variance Plot'
                            data_for_explained_variance_plot = pd.melt(
                                pca_instance.data_explained_variances,
                                id_vars="index",
                                value_vars=["explained", "cumulative"],
                            )
                            fig_variable = plot_pca_explained_variances(
                                data_to_plot=data_for_explained_variance_plot,
                                color=selectbox_variable_color,
                                template=selectbox_variable_template,
                            )
                            st.plotly_chart(
                                fig_variable,
                                theme="streamlit",
                                use_container_width=True,
                            )
                        st.markdown("**Explained Variances DataFrame**")
                        st.dataframe(
                            pca_instance.data_explained_variances.set_index(
                                keys="index"
                            ),
                            height=(
                                (len(pca_instance.data_explained_variances) + 1) * 35
                                + 3
                            ),
                        )
                    # Loadings Matrix
                    with tab_6_4_4:
                        # Create two tabs:
                        tab_6_4_4_1, tab_6_4_4_2 = st.tabs(["Heatmap", "DataFrame"])
                        with tab_6_4_4_1:
                            # Create two columns
                            col_6_1, col_6_2 = st.columns([1, 4])
                            with col_6_1:
                                st.markdown("**Plotting Options**")
                                selectbox_associations_color = st.selectbox(
                                    label="Select a color scale",
                                    options=list(AVAILABLE_COLORS_DIVERGING.keys()),
                                    index=0,
                                    key="tab_6_4_4",
                                )
                            with col_6_2:
                                fig_correlation = plot_heatmap(
                                    associations=pca_instance.data_weights,
                                    color=selectbox_associations_color,
                                    zmin=-1,
                                    zmax=1,
                                )
                                st.plotly_chart(
                                    fig_correlation,
                                    theme="streamlit",
                                    use_container_width=True,
                                )
                        with tab_6_4_4_2:
                            st.dataframe(
                                pca_instance.data_weights,
                                height=((len(pca_instance.data_weights) + 1) * 35 + 3),
                            )
                    # Transformed Data
                    with tab_6_4_5:
                        # Create two columns
                        col_6_1_1, col_6_1_2 = st.columns([1, 3])
                        with col_6_1_1:
                            data_pca_csv = convert_dataframe_to_csv(
                                pca_instance.data_pca
                            )
                            st.download_button(
                                label="Download data after PCA as CSV",
                                data=data_pca_csv,
                                file_name="data_pca.csv",
                                mime="text/csv'",
                            )
                        with col_6_1_2:
                            data_pca_xlsx = convert_dataframe_to_xlsx(
                                pca_instance.data_pca
                            )
                            st.download_button(
                                label="Download data after PCA as XLSX",
                                data=data_pca_xlsx,
                                file_name="data_pca.xlsx",
                                mime="application/vnd.ms-excel",
                            )
                        st.dataframe(
                            pca_instance.data_pca,
                            height=((len(pca_instance.data_pca) + 1) * 35 + 3),
                        )

                # Tab 6_5: Manifold
                with tab_6_5:
                    # Create two columns
                    col_6_1, col_6_2 = st.columns([1, 4])
                    with col_6_1:
                        selectbox_target = st.selectbox(
                            label="Select the target variable",
                            options=cols_cat_and_num,
                            key="tab_6_5_mani",
                        )
                        selectbox_manifold = st.selectbox(
                            label="Select the manifold implementation",
                            options=AVAILABLE_MANIFOLD_IMPLEMENTATIONS,
                        )
                        # Initiate a placeholder for the figure
                        if "fig_manifold" not in st.session_state:
                            st.session_state.fig_manifold = None

                        if st.button(
                            "Generate plot",
                            type="primary",
                            use_container_width=True,
                            key="button_plot_fig_manifold",
                        ):
                            # Compute square root of the number of observations
                            square_root_n_observations = int(sqrt(len(data)))
                            selectbox_n_neighbors = st.selectbox(
                                label="Select the number of neighbors",
                                options=range(2, 3 * square_root_n_observations),
                                index=square_root_n_observations,
                            )
                            # Initiate 'operation'
                            if selectbox_target in cols_cat:
                                operation = "classification"
                            else:
                                operation = "regression"
                            st.session_state.fig_manifold = plot_manifold(
                                data=data,
                                target_variable=selectbox_target,
                                operation=operation,
                                manifold=selectbox_manifold,
                                n_neighbors=selectbox_n_neighbors,
                            )
                    with col_6_2:
                        if st.session_state.fig_manifold is not None:
                            components.html(st.session_state.fig_manifold, height=600)


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
