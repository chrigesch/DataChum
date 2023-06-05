# Deploy Classification_and_Regression & Cluster
# ----------------------------------------------
# Editable dataframes:
# (1) https://blog.streamlit.io/editable-dataframes-are-here/?utm_medium=email&_hsmi=248366074&_hsenc=p2ANqtz--N4kRd3XhmQlvZv-ziJ_W9AkqUil8cMYedy-uJ5H16mbSNLqtdH2OrKSOlTxU_6ch9fzrxrsBb8EcVkIJTzV3iQK_ZUzgvXMlxbK1SwrFg-pIoh5U&utm_content=248366074&utm_source=hs_email # noqa: E501
# (2) https://docs.streamlit.io/library/api-reference/widgets/st.experimental_data_editor?ref=streamlit
# (3) https://data-editor.streamlit.app/?ref=streamlit

# Import moduls from local directories

# Import the required libraries
import pandas as pd
import pickle
import sklearn
import streamlit as st


def main():
    # Initiate placeholders
    pipeline_deployment = None
    label_encoder_deployment = None
    # Create two columns for uploaders
    col_1, col_2 = st.columns(2)
    with col_1:
        # Create file uploader object
        uploaded_pipeline = st.file_uploader(
            "Upload your pickled sklearn Pipeline", type=["pkl"]
        )
        if uploaded_pipeline is not None:
            # Read the file and check if it is a pipeline object
            if uploaded_pipeline.name[-3:] != "pkl":
                st.write("Type should be .PKL")
            else:
                # Read in the uploaded file
                unpickled_pipeline = pickle.load(uploaded_pipeline)
                if type(unpickled_pipeline) != sklearn.pipeline.Pipeline:
                    st.write("Only sklearn Pipelines are supported")
                else:
                    pipeline_deployment = unpickled_pipeline
    with col_2:
        # Create file uploader object
        uploaded_label_encoder = st.file_uploader(
            "Upload your pickled sklearn LabelEncoder (optional)", type=["pkl"]
        )
        if uploaded_label_encoder is not None:
            # Read the file and check if it is a pipeline object
            if uploaded_label_encoder.name[-3:] != "pkl":
                st.write("Type should be .PKL")
            else:
                # Read in the uploaded file
                unpickled_label_encoder = pickle.load(uploaded_label_encoder)
                if (
                    type(unpickled_label_encoder)
                    != sklearn.preprocessing._label.LabelEncoder
                ):
                    st.write("Only sklearn LabelEncoder are supported")
                else:
                    label_encoder_deployment = unpickled_label_encoder
    st.subheader("Deployment")

    if pipeline_deployment is not None:
        if label_encoder_deployment is None:
            st.markdown(
                "No LabelEncoder provided: Predictions will not be transformed back to original encoding"
            )
        empty_df = pd.DataFrame(
            columns=pipeline_deployment.feature_names_in_,
            index=[0],
        )
        edited_df = st.data_editor(empty_df, num_rows="dynamic")

        # Make predictions
        if "predictions" not in st.session_state:
            st.session_state.predictions = None

        button_predict = st.button(
            label="Make predictions",
            type="primary",
            use_container_width=True,
            key="button_predict",
        )
        if button_predict:
            st.session_state.predictions = pipeline_deployment(edited_df)


if __name__ == "__main__":
    # Page setup
    st.set_page_config(
        page_title="DataChum", page_icon="assets/logo_01.png", layout="wide"
    )
    main()
