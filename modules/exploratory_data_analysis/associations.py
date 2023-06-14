# Import moduls from local directories
from assets.colors import AVAILABLE_COLORS_DIVERGING
from modules.utils.preprocessing import (
    data_preprocessing,
    _get_feature_names_after_preprocessing,
)

# Import the required libraries
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import chi2_contingency, f_oneway, pearsonr, spearmanr
from sklearn.preprocessing import LabelEncoder
import streamlit as st

# Import modules for debuggung
from modules.utils.load_and_save_data import read_csv


# Calculate the correlation/strength-of-association of features in data-set with both,
# categorical and continuous features using:
# Pearson's R for continuous-continuous cases - Correlation Ratio for categorical-continuous cases  -
# Cramer's V or Theil's U for categorical-categorical cases
@st.cache_data(ttl=3600, max_entries=10)
def associations_for_categorical_and_numerical_variables(
    data: pd.DataFrame, num_num_method: str = "spearman"
):
    # Get NUMERICAL and CATEGORICAL columns
    cols_all = data.columns.to_list()
    cols_num = data.select_dtypes(include=["float", "int"]).columns.to_list()
    cols_cat = data.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.to_list()

    pipeline = data_preprocessing(
        cols_num=cols_num,
        cols_cat=cols_cat,
        imputation_numerical="most_frequent",
        scaler=None,
        imputation_categorical="most_frequent",
        one_hot_encoding=False,
    )

    data_prep = pipeline.fit_transform(data)
    # Get labels of all features
    labels = _get_feature_names_after_preprocessing(pipeline, includes_model=False)
    # Convert output to Dataframe and add columns names
    data_prep = pd.DataFrame(data_prep, columns=labels, index=data.index)
    # Initiate lists to collect all results
    results_associations_list = []
    results_pvalues_list = []
    # Loop through DataFrame and append results
    for var_1 in cols_all:
        row_dict_associations = {}
        row_dict_pvalues = {}
        for var_2 in cols_all:
            # cat_cat: Cramer's V
            if (var_1 in cols_cat) & (var_2 in cols_cat):
                result_cramers_v = cramers_v_corrected_stat(
                    data_prep[var_1], data_prep[var_2]
                )
                row_dict_associations[var_2] = result_cramers_v[0]
                row_dict_pvalues[var_2] = result_cramers_v[1]
            # num_num: Pearson or Spearman
            elif (var_1 in cols_num) & (var_2 in cols_num):
                if num_num_method == "pearson":
                    result_spearman = pearsonr(data_prep[var_1], data_prep[var_2])
                    row_dict_associations[var_2] = result_spearman[0]
                    row_dict_pvalues[var_2] = result_spearman[1]
                else:
                    result_spearman = spearmanr(data_prep[var_1], data_prep[var_2])
                    row_dict_associations[var_2] = result_spearman[0]
                    row_dict_pvalues[var_2] = result_spearman[1]
            # cat_num: Correlation Ratio
            else:
                if var_1 in cols_cat:
                    var_cat = var_1
                    var_num = var_2
                else:
                    var_cat = var_2
                    var_num = var_1
                result_eta_square_root = eta_square_root(
                    data_prep[var_cat], data_prep[var_num]
                )
                row_dict_associations[var_2] = result_eta_square_root[0]
                row_dict_pvalues[var_2] = result_eta_square_root[1]
        results_associations_list.append(row_dict_associations)
        results_pvalues_list.append(row_dict_pvalues)
    # Convert the list of dictionaries to DataFrame
    associations_df = pd.DataFrame.from_dict(results_associations_list)
    associations_df.index = cols_all
    pvalues_df = pd.DataFrame.from_dict(results_pvalues_list)
    pvalues_df.index = cols_all
    return associations_df, pvalues_df


@st.cache_data(ttl=3600, max_entries=10)
def plot_heatmap(associations, color, zmin, zmax):
    # Plot the Heatmap
    fig_correlation = px.imshow(
        associations,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale=AVAILABLE_COLORS_DIVERGING[color],
        template="simple_white",
        zmin=zmin,
        zmax=zmax,
    )
    fig_correlation.update_traces(hovertemplate="x: %{x} <br> y: %{y} <br> value: %{z}")
    return fig_correlation


def cramers_v(var1, var2):
    var1 = LabelEncoder().fit_transform(y=var1)
    var2 = LabelEncoder().fit_transform(y=var2)
    df = pd.DataFrame({"var_1": var1, "var_2": var2})
    crosstab = (
        df.groupby(["var_1", "var_2"])["var_1"]
        .count()
        .unstack(fill_value=0)
        .reset_index(drop=True)
    )
    crosstab = np.array(crosstab)
    # Finding Chi-squared test statistic and pvalue
    result = chi2_contingency(crosstab, correction=False)
    X2_stat = result[0]
    pvalue = result[1]
    # Number of observations
    obs = np.sum(np.array(crosstab))
    # Take the minimum value between the columns and the rows of the cross table
    minimum_dimension = min(crosstab.shape) - 1
    # Calculate Cramer's V
    cramers_v_score = np.sqrt(X2_stat / (obs * minimum_dimension))
    return cramers_v_score, pvalue


def cramers_v_corrected_stat(var1, var2):
    """
    Calculate Cramers V statistic for categorial-categorial association.
    uses correction from Bergsma and Wicher,
    Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    var1 = LabelEncoder().fit_transform(y=var1)
    var2 = LabelEncoder().fit_transform(y=var2)
    df = pd.DataFrame({"var_1": var1, "var_2": var2})
    crosstab = (
        df.groupby(["var_1", "var_2"])["var_1"]
        .count()
        .unstack(fill_value=0)
        .reset_index(drop=True)
    )
    confusion_matrix = np.array(crosstab)
    results_chi2 = chi2_contingency(confusion_matrix, correction=False)
    chi2 = results_chi2[0]
    pvalue = results_chi2[1]
    n = confusion_matrix.sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    minDim = min((kcorr - 1), (rcorr - 1))
    # To prevent division by zero
    if minDim == 0:
        cramers_v_score = 0
    else:
        cramers_v_score = np.sqrt(phi2corr / minDim)
    return cramers_v_score, pvalue


def eta_square_root(categorical_var, numerical_var):
    """
    Calculate the Correlation Ratio (η) for categorical-continuous association
    """
    # Perform one-way ANOVA
    groups = categorical_var.unique()
    # If categorical variable is a constant
    if len(groups) < 2:
        eta_square_root_score = 0
        pvalue = 1
        return eta_square_root_score, pvalue
    else:
        data_grouped = [numerical_var[categorical_var == group] for group in groups]
        f_value, pvalue = f_oneway(*data_grouped)
        # Calculate degrees of freedom
        df_between = len(groups) - 1
        df_within = len(categorical_var) - len(groups)
        # Calculate eta square root
        eta_square_root_score = np.sqrt(
            (f_value * df_between) / (f_value * df_between + df_within)
        )
        return eta_square_root_score, pvalue


def main():
    # Load data
    data = read_csv("data/data_c_and_r_with_missings.csv").drop("Loan_ID", axis=1)
    #    data = read_csv('data/data_c_and_r_complete.csv')
    # Compute associations
    associations_df = associations_for_categorical_and_numerical_variables(data)

    return print(associations_df.to_markdown())


if __name__ == "__main__":
    main()
