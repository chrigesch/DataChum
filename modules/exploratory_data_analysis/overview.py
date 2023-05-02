# Import moduls from local directories
from modules.exploratory_data_analysis.associations import (
    associations_for_categorical_and_numeric_variables,
)

# Import the required libraries
from collections import Counter
import numpy as np
import pandas as pd
from scipy.stats import kurtosistest, skewtest
import streamlit as st
from pyod.models.mad import MAD

# Import modules for debugging
from modules.utils.load_and_save_data import read_csv


@st.cache_data(ttl=3600, max_entries=10)
def eda_overview(data, treshold_univariate_outliers: float = 3.50):
    overview = {}
    # Get NUMERICAL, CATEGORICAL and DATETIME column names
    cols_num = data.select_dtypes(include=["float", "int"]).columns.to_list()
    cols_cat = data.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.to_list()
    cols_tim = data.select_dtypes(include=["datetime"]).columns.to_list()
    # Compute number and percentage of duplicated rows
    number_of_duplicates = len(data) - len(
        data.drop_duplicates(keep="first", inplace=False)
    )
    percentage_of_duplicates = number_of_duplicates / len(data) * 100
    # Compute number and percentage of missing values
    number_of_missing_cells = np.array(data.isna()).sum()
    percentage_of_duplicates = number_of_duplicates / len(data) * 100
    percentage_of_missing_cells = (
        number_of_missing_cells / (data.shape[0] * data.shape[1]) * 100
    )
    # Compute number and percentage of potential univariate outliers
    data_to_be_analyzed = data[cols_num].dropna()
    number_of_outliers = 0
    for column in cols_num:
        outlier_labels = (
            MAD(threshold=treshold_univariate_outliers)
            .fit(np.array(data_to_be_analyzed[column]).reshape([-1, 1]))
            .labels_
        )
        number_of_outliers += len(outlier_labels[(outlier_labels == 1)])
    percentage_of_outliers = number_of_outliers / (data.shape[0] * data.shape[1]) * 100
    # Fill in the dictionary
    overview["Number of observations"] = data.shape[0] * data.shape[1] - (
        np.array(data.isna()).sum()
    )
    overview["Number of rows"] = data.shape[0]
    overview["Number of variables"] = data.shape[1]
    overview["Number of categorical variables"] = len(cols_cat)
    overview["Number of numerical variables"] = len(cols_num)
    overview["Number of date or datetime variables"] = len(cols_tim)
    overview["Number of missing cells"] = number_of_missing_cells
    overview["Missing cells (%)"] = "{0:.1f}".format(percentage_of_missing_cells) + "%"
    overview["Number of duplicated rows"] = number_of_duplicates
    overview["Duplicated rows (%)"] = "{0:.1f}".format(percentage_of_duplicates) + "%"
    overview["Number of univariate outliers"] = number_of_outliers
    overview["Univariate outliers (%)"] = "{0:.1f}".format(percentage_of_outliers) + "%"
    overview_df = pd.DataFrame(overview, index=[0]).T
    overview_df.columns = [""]
    return overview_df


@st.cache_data(ttl=3600, max_entries=10)
def eda_overview_detailed(
    data,
    associations_df,
    threshold_cardinality: int = 50,
    threshold_correlation: float = 0.5,
    treshold_kurtosistest: float = 0.05,
    threshold_imbalanced: float = 0.95,
    treshold_skewtest: float = 0.05,
    treshold_univariate_outliers: float = 3.50,
):
    # Get ALL, NUMERICAL, CATEGORICAL and DATETIME column names
    cols_all = data.columns.to_list()
    cols_num = data.select_dtypes(include=["float", "int"]).columns.to_list()
    cols_cat = data.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.to_list()
    cols_tim = data.select_dtypes(include=["datetime"]).columns.to_list()

    # Initiate lists to collect the results
    list_alerts = []
    list_details = []
    list_variable = []

    # Alerts that are valid for all columns
    for column in cols_all:
        # Analyze whether column only contains one value
        if len(data[column].unique()) == 1:
            list_variable.append(column)
            list_alerts.append("Constant")
            list_details.append(
                "Column only contains one value: " + str(data[column].unique())
            )
        # Analyze whether correlations (either Spearman, Cramer, Pearson, Kendall, 𝜙k)
        # are above the warning threshold (configurable).
        associations_above_threshold = associations_df[
            (associations_df[column] > threshold_correlation)
            | (associations_df[column] < -threshold_correlation)
        ][column]
        if len(associations_above_threshold) > 0:
            for row in associations_above_threshold.items():
                # If column name is not the same as row name
                if column != row[0]:
                    list_variable.append(column)
                    list_alerts.append("High Correlation")
                    list_details.append(
                        "Highly correlated with "
                        + str(row[0])
                        + " "
                        + "{0:.2f}".format(row[1])
                    )
        # Analyze whether column has missing values
        number_of_missing_values = np.array(data[column].isna()).sum()
        percentage_of_missing_values = (
            number_of_missing_values / len(data[column]) * 100
        )
        if number_of_missing_values > 0:
            list_variable.append(column)
            list_alerts.append("Missing Values")
            list_details.append(
                "Column has "
                + str(number_of_missing_values)
                + " missing values ("
                + "{0:.1f}".format(percentage_of_missing_values)
                + "%)"
            )

    # Alerts that are only valid for CATEGORICAL variables
    for column in cols_cat:
        # Analyze whether the column has more than 50 distinct values
        if len(data[column].unique()) > threshold_cardinality:
            list_variable.append(column)
            list_alerts.append("Cardinality")
            list_details.append(
                "Column has " + str(len(data[column].unique())) + " distinct values"
            )
        # Analyze whether the column is highly imbalanced.
        shannon_entropy = _shannon_entropy_balance(data[column])
        if shannon_entropy < threshold_imbalanced:
            list_variable.append(column)
            list_alerts.append("Imbalanced")
            list_details.append(
                "Highly imbalanced (shannon entropy="
                + "{0:.2f}".format(shannon_entropy)
                + ")"
            )
        # Analyze whether all values of the column are unique (count of unique values equals column’s length)
        if len(data[column]) == len(data[column].unique()):
            list_variable.append(column)
            list_alerts.append("Unique Values")
            list_details.append("All values of the column are unique")

    # Alerts that are only valid for NUMERICAL variables
    for column in cols_num:
        # Analye whether the column presents kurtosis
        p_value_kurtosistest = kurtosistest(data[column])
        if p_value_kurtosistest[1] < treshold_kurtosistest:
            list_variable.append(column)
            list_alerts.append("Kurtosis")
            list_details.append(
                "Not mesokurtic (statistic="
                + "{0:.2f}".format(p_value_kurtosistest[0])
                + ", p="
                + "{0:.2f}".format(p_value_kurtosistest[1])
                + ")"
            )
        # Analyze whether columns contain infinite values
        number_of_infinite_values = np.isinf(data[column]).values.sum()
        percentage_of_infinite_values = (
            number_of_infinite_values / len(data[column]) * 100
        )
        if number_of_missing_values > 0:
            list_variable.append(column)
            list_alerts.append("Infinite Values")
            list_details.append(
                "Column has "
                + str(number_of_missing_values)
                + " missing values ("
                + "{0:.1f}".format(percentage_of_infinite_values)
                + "%)"
            )
        # Analye whether the column presents skewness
        p_value_skewtest = skewtest(data[column])
        if p_value_skewtest[1] < treshold_skewtest:
            list_variable.append(column)
            list_alerts.append("Skewness")
            list_details.append(
                "Highly skewed (statistic="
                + "{0:.2f}".format(p_value_skewtest[0])
                + ", p="
                + "{0:.2f}".format(p_value_skewtest[1])
                + ")"
            )
        # Analyze whether the column has potential univariate outliers
        data_to_be_analyzed = data[cols_num].dropna()
        number_of_outliers = 0
        outlier_labels = (
            MAD(threshold=treshold_univariate_outliers)
            .fit(np.array(data_to_be_analyzed[column]).reshape([-1, 1]))
            .labels_
        )
        number_of_outliers += len(outlier_labels[(outlier_labels == 1)])
        if number_of_outliers > 0:
            list_variable.append(column)
            list_alerts.append("Outliers")
            list_details.append(
                "Column has "
                + "{0:.0f}".format(number_of_outliers)
                + " observations that are more than "
                + "{0:.1f}".format(treshold_univariate_outliers)
                + " Median Absolute Deviations (MAD) away from the mean"
            )

    # Alerts that are only valid for DATETIME variables
    if len(cols_tim) > 0:
        for column in cols_tim:
            list_variable.append(column)
            list_alerts.append("Date")
            list_details.append("Column contains Date or Datetime records")

    # Initiate a Dataframe that contains all results
    results = pd.DataFrame()
    results["Variable"] = list_variable
    results["Alert"] = list_alerts
    results["Detail"] = list_details
    results = results.sort_values(by="Alert").reset_index(drop=True)
    return results


######################################
# Private Methods / Helper functions #
######################################
# https://stats.stackexchange.com/questions/239973/a-general-measure-of-data-set-imbalance#:~:text=On%20a%20data%20set%20of,data%20set%20is%20very%20unbalanced
def _shannon_entropy_balance(seq):
    n = len(seq)
    classes = [(clas, float(count)) for clas, count in Counter(seq).items()]
    k = len(classes)

    H = -sum(
        [(count / n) * np.log((count / n)) for clas, count in classes]
    )  # shannon entropy
    return H / np.log(k)


#################
# For debugging #
#################


def main():
    # Load data
    #    data = read_csv('data/data_c_and_r_with_missings.csv').drop('Loan_ID', axis=1)
    data = read_csv("data/data_c_and_r_complete.csv")
    # Compute associations
    associations_df = associations_for_categorical_and_numeric_variables(data)

    return print(eda_overview(data)), print(
        eda_overview_detailed(data, associations_df)
    )


#    return print(eda_overview(data).to_markdown()), print(eda_overview_detailed(data, associations_df).to_markdown())

if __name__ == "__main__":
    main()
