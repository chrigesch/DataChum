# Import moduls from local directories
from pages_ import (
    Classification_and_Regression,
    Cluster_Analysis,
    Deployment,
    Exploratory_Data_Analysis,
)
from modules.utils.load_and_save_data import (
    read_csv,
    read_xlsx,
)

# Import the required libraries
from PIL import Image
import streamlit as st


def main():
    # Page setup
    st.set_page_config(
        page_title="DataChum", page_icon="assets/logo_01.png", layout="wide"
    )

    # Display the sidebar with a menu of apps
    with st.sidebar:
        # Add a title and intro text
        st.title("Data Chum")
        st.markdown("Your web app to analyze data")

        # Import image and add it to the sidebar
        image = Image.open("assets/logo_01.png")
        st.image(image, width=150)
        st.markdown("**Select a module from the list below**")

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
        button_cluster = st.button(
            label="**Cluster Analysis**",
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
    if button_cluster:
        st.session_state.page = Cluster_Analysis
    if button_deployment:
        st.session_state.page = Deployment

    # Add a DATA file uploader object (Deployment needs "model" and (optionally) "label_encoder")
    if st.session_state.page != Deployment:
        # Set placeholder for data
        if "data" not in st.session_state:
            st.session_state.data = None
        if "data_updated" not in st.session_state:
            st.session_state.data_updated = False
        # Create file uploader object
        uploaded_file = st.file_uploader("Upload your database", type=["csv", "xlsx"])
        if (uploaded_file is not None) & (st.session_state.data_updated is False):
            # Read the file to a dataframe using pandas
            if uploaded_file.name[-3:] == "csv":
                # Read in the csv file
                st.session_state.data = read_csv(uploaded_file)
            elif uploaded_file.name[-4:] == "xlsx":
                # Read in the csv file
                st.session_state.data = read_xlsx(uploaded_file)
            else:
                st.write("Type should be .CSV or .XLSX")

    st.session_state.page.main()


if __name__ == "__main__":
    main()
