# Import moduls from local directory
from modules.utils.load_and_save_data import read_csv

# Import the required libraries
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss

data = read_csv('../data/data_timeserie.csv')

cols_tim = data.select_dtypes(include = ['datetime']).columns.to_list()

data_to_be_analyzed = data.set_index(cols_tim, drop = True)

# Resample timeseries to hourly (if necesary) / Otra Alternativa: analysis_features.asfreq(freq = 'H')
#data_to_be_analyzed = data_to_be_analyzed.resample('H').sum()

decompose_result_mult = seasonal_decompose(data_to_be_analyzed, model="multiplicative")

kpss(data_to_be_analyzed)

adfuller(data_to_be_analyzed)