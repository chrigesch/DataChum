# Import moduls from local directories
from assets.colors import AVAILABLE_COLORS_DIVERGING

# Import the required libraries
from dython.nominal import associations
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import chi2_contingency
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
    crosstab = pd.crosstab(
        var1,
        var2,
        rownames=None,
        colnames=None,
    ).reset_index()
    # Check if confusion matrix is 2x2 to use a correction or no
    if crosstab.shape[0] == 2:
        correct = False
    else:
        correct = True
    # Finding Chi-squared test statistic,
    X2_stat = chi2_contingency(crosstab, correction=correct)[0]
    # Number of observations
    obs = np.sum(np.array(crosstab))
    # Take the minimum value between the columns and the rows of the cross table
    minimum_dimension = min(crosstab.shape) - 1
    # Calculate Cramer's V
    return np.sqrt(X2_stat / (obs * minimum_dimension))


def main():
    # Load data
    data = read_csv("data/data_c_and_r_with_missings.csv").drop("Loan_ID", axis=1)
    #    data = read_csv('data/data_c_and_r_complete.csv')
    # Compute associations
    associations_df = associations_for_categorical_and_numeric_variables(data)

    return print(associations_df.to_markdown())


if __name__ == "__main__":
    main()
