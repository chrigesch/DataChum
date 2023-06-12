# Import moduls from local directory

# Import the required libraries
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.writer.excel import save_virtual_workbook
import pandas as pd
import streamlit as st

# streamlit 1.8 vs 1.19
# https://docs.streamlit.io/knowledge-base/dependencies/libgl


@st.cache_data(ttl=3600, max_entries=10)
def convert_dataframe_to_csv(dataframe):
    return dataframe.to_csv().encode("utf-8")


@st.cache_data(ttl=3600, max_entries=10)
def convert_dataframe_to_xlsx(dataframe):
    # Create a workbook and a worksheet
    workbook = Workbook()
    worksheet = workbook.active
    # Loop through rows and append them to worksheet
    for r in dataframe_to_rows(dataframe, index=True, header=True):
        worksheet.append(r)
    return save_virtual_workbook(workbook)


@st.cache_data(ttl=3600, max_entries=10)
def read_csv(data, *args, **kwargs):
    """Drop-in replacement for Pandas pd.read_csv. It invokes
    pd.read_csv() (passing its arguments) and then auto-
    matically detects and converts each column whose datatype
    is 'object' to a datetime just when ALL of the column's
    non-NaN values can be successfully parsed by
    pd.to_datetime(), and returns the resulting dataframe.
    """
    data = _dt_inplace(pd.read_csv(data, *args, **kwargs))
    data = _reduce_memory_usage(data)
    return data


@st.cache_data(ttl=3600, max_entries=10)
def read_xlsx(data, *args, **kwargs):
    """Drop-in replacement for Pandas pd.read_excel. It invokes
    pd.read_csv() (passing its arguments) and then auto-
    matically detects and converts each column whose datatype
    is 'object' to a datetime just when ALL of the column's
    non-NaN values can be successfully parsed by
    pd.to_datetime(), and returns the resulting dataframe.
    """
    data = _dt_inplace(pd.read_excel(data, *args, **kwargs, engine="openpyxl"))
    data = _reduce_memory_usage(data)
    return data


######################################
# Private Methods / Helper functions #
######################################


def _dt_inplace(df):
    """Automatically detect and convert (in place!) each
    dataframe column of datatype 'object' to a datetime just
    when ALL of its non-NaN values can be successfully parsed
    by pd.to_datetime().  Also returns a ref. to df for
    convenient use in an expression.
    """
    from pandas.errors import ParserError

    for c in df.columns[df.dtypes == "object"]:  # don't cnvt num
        try:
            df[c] = pd.to_datetime(df[c])
        except (ParserError, TypeError, ValueError):  # Can't cnvrt some
            pass  # ...so leave whole column as-is unconverted
    return df


@st.cache_data(ttl=3600, max_entries=10)
def _reduce_memory_usage(data):
    cols_to_change = data.select_dtypes(include=["float64"]).columns.to_list()
    for col_name in cols_to_change:
        data[col_name] = data[col_name].astype("float32")
    cols_to_change = data.select_dtypes(include=["int64"]).columns.to_list()
    for col_name in cols_to_change:
        data[col_name] = data[col_name].astype("int32")
    return data
