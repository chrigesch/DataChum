# Import moduls from local directories
from assets.colors import AVAILABLE_COLORS_DIVERGING

# Import the required libraries
from dython.nominal import associations
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import chi2_contingency, f_oneway
from sklearn.preprocessing import LabelEncoder
import streamlit as st

# Import modules for debuggung
from modules.utils.load_and_save_data import read_csv


# Calculate the correlation/strength-of-association of features in data-set with both,
# categorical and continuous features using:
# Pearson's R for continuous-continuous cases - Correlation Ratio for categorical-continuous cases  -
# Cramer's V or Theil's U for categorical-categorical cases
@st.cache_data(ttl=3600, max_entries=10)
def associations_for_categorical_and_numeric_variables(data):
    return associations(data, compute_only=True, plot=False)["corr"]


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
    # Check if confusion matrix is 2x2 to use a correction or no
    if crosstab.shape[0] == 2:
        correct = False
    else:
        correct = True
    # Finding Chi-squared test statistic and pvalue
    result = chi2_contingency(crosstab, correction=correct)
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
    print(crosstab.shape)
    print(crosstab)
    print(confusion_matrix)
    print(crosstab.values)
    results_chi2 = chi2_contingency(confusion_matrix)
    chi2 = results_chi2[0]
    pvalue = results_chi2[1]
    n = confusion_matrix.sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    cramers_v_score = np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))
    return cramers_v_score, pvalue


def eta_square_root(categorical_var, numerical_var):
    """
    Calculate the Correlation Ratio (η) for categorical-continuous association
    """
    # Perform one-way ANOVA
    groups = categorical_var.unique()
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
    associations_df = associations_for_categorical_and_numeric_variables(data)

    return print(associations_df.to_markdown())


if __name__ == "__main__":
    main()
