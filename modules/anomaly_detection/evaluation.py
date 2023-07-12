# Import moduls from local directories
from assets.colors import get_color

# Import the required Libraries
import plotly.express as px
from plotly.subplots import make_subplots


def plot_anomalies_evaluation(
    outlier_scores,
    name_model: str,
    color: str,
):
    # Create a list of colors for the markers
    list_colors = get_color(color, 1)[0]
    fig = make_subplots(
        rows=2, cols=1, specs=[[{"secondary_y": True}], [{"secondary_y": True}]]
    )
    fig.add_trace(
        px.ecdf(
            outlier_scores,
            markers=True,
            lines=False,
            color_discrete_sequence=[list_colors],
        ).data[0],
        row=2,
        col=1,
        secondary_y=False,
    )

    fig.add_trace(
        px.histogram(outlier_scores, opacity=0.2, histnorm="percent").data[0],
        row=2,
        col=1,
        secondary_y=True,
    )

    fig.add_trace(
        px.box(x=outlier_scores, color_discrete_sequence=[list_colors]).data[0],
        row=1,
        col=1,
        secondary_y=False,
    )
    # Set y-axes titles
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_yaxes(
        title_text="ECDF - probability", row=2, col=1, secondary_y=False, range=[0, 1.1]
    )
    fig.update_yaxes(
        title_text="Histogram - percent", row=2, col=1, secondary_y=True, showgrid=False
    )
    fig.update_layout(title_text=name_model, showlegend=False)
    return fig
