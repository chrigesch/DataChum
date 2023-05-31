from PIL import Image
import streamlit as st
from pages_ import Classification_and_Regression, Deployment, Exploratory_Data_Analysis

# Page setup
st.set_page_config(page_title="DataChum", page_icon="assets/logo_01.png", layout="wide")

# Display the sidebar with a menu of apps
with st.sidebar:
    # Add a title and intro text
    st.title("Data Chum")
    st.markdown("Your web app to analyze data")

    # Import image and add it to the sidebar
    image = Image.open("assets/logo_01.png")
    st.image(image, use_column_width=True)
    st.markdown("**Select an application from the list below**")

    # Create a button for each application
    button_exploratory_data_analysis = st.button(
        label="**Exploratory Data Analysis**",
        type="secondary",
        use_container_width=True,
    )
    button_classification_and_regression = st.button(
        label="**Classification and Regression**",
        type="secondary",
        use_container_width=True,
    )
    button_deployment = st.button(
        label="**Deployment**", type="secondary", use_container_width=True
    )

# Set Default page | Session state variables
if "page" not in st.session_state:
    st.session_state.page = Exploratory_Data_Analysis

# Run the chosen app
if button_exploratory_data_analysis:
    st.session_state.page = Exploratory_Data_Analysis
if button_classification_and_regression:
    st.session_state.page = Classification_and_Regression
if button_deployment:
    st.session_state.page = Deployment

st.session_state.page.main()
