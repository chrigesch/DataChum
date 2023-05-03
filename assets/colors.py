# Import the required libraries
import plotly.express as px
import seaborn as sns

# https://www.w3schools.com/colors/colors_names.asp
# px.colors.cyclical.swatches()
# px.colors.diverging.swatches()
# px.colors.sequential.swatches()
# px.colors.qualitative.swatches()

AVAILABLE_COLORS_DIVERGING = {
    "Armyrose": px.colors.diverging.Armyrose,
    "Earth": px.colors.diverging.Earth,
    "Spectral": px.colors.diverging.Spectral,
    "Tropic": px.colors.diverging.Tropic,
    "BlueRedDark": ["blue", "royalblue", "black", "firebrick", "darkred"],
    "BlueRedLight": ["blue", "royalblue", "white", "firebrick", "darkred"],
    "GreenOrangeDark": ["darkcyan", "lightseagreen", "black", "coral", "orangered"],
    "GreenOrangeLight": ["darkcyan", "lightseagreen", "white", "coral", "orangered"],
    "PurpleYellowDark": ["rebeccapurple", "slateblue", "black", "khaki", "yellow"],
    "PurpleYellowLight": ["rebeccapurple", "slateblue", "white", "khaki", "yellow"],
}

AVAILABLE_COLORS_SEQUENTIAL = (
    "hot",
    "hsv",
    "icefire",
    "magma",
    "plasma",
    "rainbow",
    "twilight",
    "viridis",
)


# Extract a list of colors for later use with each visualization
def get_color(name: str, number: int):
    pal = list(sns.color_palette(palette=name, n_colors=number).as_hex())
    return pal
