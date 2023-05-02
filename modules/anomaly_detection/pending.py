# (1) Add "analysis" (a) which method and (b) which treshold is the best to predict a specific target variable
#    (b) Create a 'tuneable' sklearnmodel
# (2) Add "Anomaly detection" in CV? 
# ------------------------------------------------------------------------------------------------------------
# (1) https://towardsdatascience.com/how-to-perform-multivariate-outlier-detection-in-python-pyod-for-machine-learning-b0a9c557a21c
# (2) https://medium.com/datasparq-technology/anomaly-detection-for-beginners-640c949d206
# (3) https://towardsdatascience.com/5-ways-to-detect-outliers-that-every-data-scientist-should-know-python-code-70a54335a623
# (4) https://towardsdatascience.com/outlier-detection-with-isolation-forest-3d190448d45e
# 

# Import moduls from local directories
from modules.utils.preprocessing import data_preprocessing, _get_feature_names_after_preprocessing

# Import the required libraries
import pandas as pd
import streamlit as st
from yellowbrick.regressor import CooksDistance


# ONLY for 'regression'
@st.cache_data(ttl=3600, max_entries=10)
def plot_cooks_distance(data, target_variable):
    # Drop all rows with NaNs in target_Variable
    data = data.dropna(subset=target_variable)
    # Separate X and y
    y = pd.Series(data[target_variable])
    X = data.drop(target_variable, axis=1)
    # Get NUMERICAL and CATEGORICAL columns
    cols_num = X.select_dtypes(include = ['float', 'int']).columns.to_list()
    cols_cat = X.select_dtypes(include = ['object', 'category','bool']).columns.to_list()
    # Create preprocessing pipeline and fit_transform it
    pipeline = data_preprocessing(cols_num, cols_cat,
                                  imputation_numeric='mean', scaler='zscore',
                                  imputation_categorical='most_frequent',
                                  one_hot_encoding=True)
    X_prep = pipeline.fit_transform(X)
    # Get labels of all features
    labels = _get_feature_names_after_preprocessing(pipeline,includes_model=False)
    # Convert output to Dataframe and add columns names
    X_prep = pd.DataFrame(X_prep, columns=labels, index=X.index)
    # Instantiate and fit the visualizer    
    visualizer = CooksDistance()
    visualizer.fit(X_prep, y)
    return visualizer
#    return visualizer.show()
