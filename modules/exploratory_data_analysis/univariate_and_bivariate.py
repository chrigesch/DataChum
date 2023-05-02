# Import moduls from local directories
from assets.colors import get_color

# Import the required libraries
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objs as go
from statsmodels.graphics.gofplots import qqplot
import streamlit as st

# Import libraries for debugging
from modules.utils.load_and_save_data import read_csv

AVAILABLE_TEMPLATES = [
    "gridon",
    "plotly",
    "plotly_dark",
    "plotly_white",
    "xgridoff",
    "ygridoff",
]


# Univariate analysis
@st.cache_data(ttl=3600, max_entries=10)
def plot_cat(
    data, var_cat: str, var_num: str, plot_type: str, color: str, template: str
):
    # Create plot
    if plot_type == "Bar":
        fig_variable = px.bar(
            data,
            x=var_cat,
            y=var_num,
            color=var_cat,
            color_discrete_sequence=get_color(color, len(data)),
            template=template,
        )
        fig_variable.update_layout(xaxis_type="category")
    # Create plot
    elif (plot_type == "Donut") | (plot_type == "Pie"):
        fig_variable = px.pie(
            data,
            values=var_num,
            names=var_cat,
            color=var_cat,
            color_discrete_sequence=get_color(color, len(data)),
            template=template,
        )
        fig_variable.update_layout(xaxis_type="category")
        if plot_type == "Donut":
            fig_variable.update_traces(hole=0.3)
    # Create plot
    elif plot_type == "Polar":
        fig_variable = px.bar_polar(
            data,
            theta=var_cat,
            r=var_num,
            color=var_num,
            color_continuous_scale=get_color(color, len(data)),
            template=template,
        )
        fig_variable.update_polars(angularaxis_type="category")
    # Create plot
    elif plot_type == "Radar":
        fig_variable = px.line_polar(
            data,
            theta=var_cat,
            r=var_num,
            line_close=True,
            color_discrete_sequence=[get_color(color, len(data))[0]],
            template=template,
        )
        fig_variable.update_traces(fill="toself")
        fig_variable.update_polars(angularaxis_type="category")
    # Create plot
    elif plot_type == "Treemap":
        fig_variable = px.treemap(
            data,
            path=[px.Constant(var_cat), var_cat],
            values=data[var_num],
            color=data[var_num],
            color_continuous_scale=get_color(color, len(data)),
            template=template,
        )

    fig_variable.update_layout(width=600, height=400)
    return fig_variable


@st.cache_data(ttl=3600, max_entries=10)
def plot_num(
    data, var_num: str, var_cat: str, plot_type: str, color: str, template: str
):
    # Create plot
    if plot_type in ["Histogram", "KDE", "Histogram_KDE"]:
        if var_cat is None:
            list_data = [data[var_num].dropna()]
            list_labels = ["Density"]
            list_colors = get_color(color, len(list_labels))
        else:
            list_data = []
            list_labels = []
            for value in data[var_cat].dropna().unique():
                list_data.append(data[data[var_cat] == value][var_num].dropna())
                list_labels.append(value)
            list_colors = get_color(color, len(list_labels))
        # Freedman–Diaconis rule to compute bin width
        q1 = data[var_num].dropna().quantile(0.25)
        q3 = data[var_num].dropna().quantile(0.75)
        iqr = q3 - q1
        bin_width = (2 * iqr) / (len(data[var_num].dropna()) ** (1 / 3))
        if plot_type == "Histogram":
            fig_variable = ff.create_distplot(
                list_data,
                bin_size=bin_width,
                show_curve=False,
                colors=list_colors,
                group_labels=list_labels,
            )
        elif plot_type == "KDE":
            fig_variable = ff.create_distplot(
                list_data,
                bin_size=bin_width,
                show_hist=False,
                colors=list_colors,
                group_labels=list_labels,
            )
        else:
            fig_variable = ff.create_distplot(
                list_data,
                bin_size=bin_width,
                colors=list_colors,
                group_labels=list_labels,
            )
    # Create plot
    if plot_type == "QQ-Plot":
        # Draw the standardized line
        qqplot_data = qqplot(data[var_num], line="s").gca().lines
        fig_variable = go.Figure()
        fig_variable.add_trace(
            {
                "type": "scatter",
                "x": qqplot_data[0].get_xdata(),
                "y": qqplot_data[0].get_ydata(),
                "mode": "markers",
                "marker": {"color": get_color(color, 2)[0]},
            }
        )
        fig_variable.add_trace(
            {
                "type": "scatter",
                "x": qqplot_data[1].get_xdata(),
                "y": qqplot_data[1].get_ydata(),
                "mode": "lines",
                "line": {"color": get_color(color, 2)[1]},
            }
        )
        fig_variable["layout"].update(
            {
                "title": "Quantile-Quantile Plot",
                "xaxis": {"title": "Theoritical Quantities", "zeroline": False},
                "yaxis": {"title": "Sample Quantities", "zeroline": False},
                "showlegend": False,
            }
        )
    # Create plot
    if plot_type == "Box-Plot":
        if var_cat is None:
            list_colors = get_color(color, 1)
        else:
            list_colors = get_color(color, len(data[var_cat].dropna().unique()))
        fig_variable = px.box(
            data,
            x=var_cat,
            y=var_num,
            color_discrete_sequence=list_colors,
            color=var_cat,
            template=template,
        )
        fig_variable.update_layout(xaxis_type="category")
        fig_variable.update_layout(width=400, height=500)

    if plot_type != "Box-Plot":
        fig_variable.update_layout(width=600, height=400)
    fig_variable.update_layout(template=template)
    return fig_variable


# Bivariate analysis
@st.cache_data(ttl=3600, max_entries=10)
def plot_cat_cat(
    data, target: str, feature: str, barmode: str, color: str, template: str
):
    data_to_be_plotted = pd.DataFrame(
        data[[target, feature]].value_counts()
    ).reset_index()
    data_to_be_plotted.columns = [target, feature, "Count"]
    if barmode in ["group", "stack"]:
        fig_variable = px.bar(
            data_to_be_plotted.sort_values(by=target),
            x=target,
            y="Count",
            color=feature,
            barmode=barmode,
            color_discrete_sequence=get_color(color, len(data[feature].value_counts())),
            template=template,
        )
    elif barmode == "100_stack":
        fig_variable = px.histogram(
            data_to_be_plotted,
            x=target,
            y="Count",
            color=feature,
            barnorm="percent",
            text_auto=".2f",
            color_discrete_sequence=get_color(color, len(data[feature].value_counts())),
            template=template,
        )
    fig_variable.update_layout(xaxis_type="category")
    fig_variable.update_layout(width=600, height=400)
    return fig_variable


@st.cache_data(ttl=3600, max_entries=10)
def plot_num_num(
    data,
    target: str,
    feature: str,
    plot_type: str,
    trendline: str,
    marginal_x,
    marginal_y,
    color: str,
    template: str,
):
    list_colors = get_color(color, len(data[feature].value_counts()))
    if (plot_type == "scatter") & (trendline == "lowess"):
        fig_variable = px.scatter(
            data,
            x=feature,
            y=target,
            marginal_x=marginal_x,
            marginal_y=marginal_y,
            trendline=trendline,
            trendline_options=dict(frac=0.5),
            trendline_color_override=list_colors[-1],
            color_discrete_sequence=list_colors,
            template=template,
        )
    else:
        fig_variable = px.scatter(
            data,
            x=feature,
            y=target,
            marginal_x=marginal_x,
            marginal_y=marginal_y,
            trendline=trendline,
            trendline_color_override=list_colors[-1],
            color_discrete_sequence=list_colors,
            template=template,
        )
    return fig_variable


#################
# For debugging #
#################


def main():
    data = read_csv(
        "/mnt/d/Proyectos/DataChum/data/data_c_and_r_with_missings.csv"
    ).drop("Loan_ID", axis=1)

    figure = plot_num(
        data,
        var_num="CoapplicantIncome",
        var_cat="Dependents",
        plot_type="Histogram",
        color="crest",
        template="gridon",
    )

    return figure.show()


if __name__ == "__main__":
    main()
