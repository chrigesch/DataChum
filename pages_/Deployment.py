# Import the required libraries
import pandas as pd
from pandas.errors import ParserError
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
                st.error("Type should be .PKL")
            else:
                # Read in the uploaded file
                unpickled_pipeline = pickle.load(uploaded_pipeline)
                if type(unpickled_pipeline) != sklearn.pipeline.Pipeline:
                    st.error("Only sklearn Pipelines are supported")
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
                st.error("Type should be .PKL")
            else:
                # Read in the uploaded file
                unpickled_label_encoder = pickle.load(uploaded_label_encoder)
                if (
                    type(unpickled_label_encoder)
                    != sklearn.preprocessing._label.LabelEncoder
                ):
                    st.error("Only sklearn LabelEncoder are supported")
                else:
                    label_encoder_deployment = unpickled_label_encoder
    st.subheader("Deployment")

    if pipeline_deployment is not None:
        st.markdown(
            "To provide data you can (a) insert data manually or (b) copy data from "
            "another sheet document to clipboard, then select any cell "
            "of the table below and paste it in (via ctrl/cmd + v)."
        )
        if label_encoder_deployment is None:
            st.markdown(
                "No LabelEncoder provided: Predictions will not be transformed back to original encoding"
            )
        empty_df = pd.DataFrame(
            columns=pipeline_deployment.feature_names_in_,
            index=[0],
        )
        edited_df = st.data_editor(
            empty_df,
            num_rows="dynamic",
            hide_index=False,
        )
        st.warning(
            str(len(edited_df))
            + " row(s) were recognized. Make sure that every row has its unique index number."
        )
        button_predict = st.button(
            label="Make predictions",
            type="primary",
            use_container_width=True,
            key="button_predict",
        )
        if button_predict:
            # Convert all possible columns to numeric (st.data_editor creates "object")
            for col in edited_df.columns[edited_df.dtypes == "object"]:
                try:
                    edited_df[col] = pd.to_numeric(edited_df[col], errors="coerce")
                except (ParserError, ValueError):
                    pass  # ...so leave whole column as-is unconverted
            # Make predictions and transform labels, if encoder was provided
            predictions = pipeline_deployment.predict(edited_df)
            if label_encoder_deployment is not None:
                predictions = label_encoder_deployment.inverse_transform(predictions)
            st.markdown("Predictions of " + str(pipeline_deployment.steps[-1][1]))
            st.dataframe(predictions, use_container_width=False)


if __name__ == "__main__":
    # Page setup
    st.set_page_config(
        page_title="DataChum", page_icon="assets/logo_01.png", layout="wide"
    )
    main()
