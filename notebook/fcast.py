# A module contain functions used for forecasting
import os
import numpy as np
import pandas as pd
import seaborn as sns
import datetime
import impyute as impy
import pmdarima as pm

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
from   matplotlib.ticker import FormatStrFormatter, ScalarFormatter

from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.stattools import grangercausalitytests as granger
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.api import ExponentialSmoothing, VAR, VECM
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tools.eval_measures import rmse, aic, bic
from statsmodels.tools.eval_measures import meanabs as mae
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from types import SimpleNamespace

import warnings
import joblib
warnings.filterwarnings("ignore")
pd.options.display.float_format = '{:,.4f}'.format

# import conversion dict
monthTH = pd.read_excel('../data/month.xlsx')
month_map = dict(zip(monthTH.short_th, monthTH.num))

# define functions

def read_price(name):
    path = f'../data/{name}.xls'
    df = (pd.read_excel(path, sheet_name = 'price', skiprows = 2, na_values = '-')
          .melt(id_vars = 'ปี', var_name = 'month', value_name='actual')
          .rename(columns = {'ปี':'year'})
          .assign(year = lambda x: x.year-543)
          .assign(month = lambda x: x.month.str.strip().map(month_map))
          .assign(actual = lambda x: pd.to_numeric(x.actual, errors='coerce'))
          .assign(date = lambda x: pd.to_datetime(
              '1' + '/' + x.month.astype(str) + '/' + x.year.astype(str),
              dayfirst = True,))
          .query('year<2020 & ~actual.isna()')
          .set_index('date')
          .sort_index()
          .asfreq(freq='MS')
          .drop(columns=['year', 'month'])
          .interpolate(method='time')
       )
    return df

def test_stationary(df, name=None, col='actual'):
    """
    Stationary test of a series
    Input: A dataframe with a specific column
    """
    if name == None:
        name = df.meta.shortname
    col_names = ['price', 'level', 'adf_stat', 'adf_pval', 'kpss_stat', 'kpss_pval']
    level = df[col]
    diff1 = df[col].diff().dropna()

    result = []
    result.append([name, 'level', *adfuller(level)[:2], *kpss(level)[:2]])
    result.append([name, 'diff1', *adfuller(diff1)[:2], *kpss(diff1)[:2]])

    return pd.DataFrame(result, columns=col_names).set_index(['price', 'level']).round(4)


def series2supervised(ser, n_lags=12, n_fcast=6):
    """
    Input: a series or a dataframe with a time index and has n observations
    Output: n x (1+n_lag+n_fcast) dataframe
    """
    df = pd.DataFrame(columns=[f'x{i}' for i in range(n_lags)] + [f'y{i}' for i in range(1, n_fcast+1)],
                      index=ser.index)
    for i in range(n_lags):
        df[f'x{i}'] = ser.shift(i)
    for i in range(1, n_fcast+1):
        df[f'y{i}'] = ser.shift(-i)
    return df

def split_train_test(df, end_train='2012-12'):
    train = df.sup_scaled.loc[:end_train].dropna()
    test  = df.sup_scaled.loc[end_train:][1:]
    return train, test

def data_transform(df):
    """
    input = a dataframe with nx1 or a series with n observations
    output = 2 tuple where (1) scaler object for inverse transform (2) the scaled dataframe with nx1
    log -> diff -> scale
    """
    if df.ndim == 1:
        df = pd.DataFrame(df)
    logged   = np.log(df)
    log_diff = logged.diff()
    scaler   = StandardScaler()
    scaled   = scaler.fit_transform(log_diff) #expect 2D array
    return scaler, pd.DataFrame(scaled, columns=['scaled'], index=df.index)

def data_inverse(fcast, actual, scaler, base_index):
    """
    input = a numpy array 
    output = hx1 array of inverse series
    inverse scale -> inverse diff -> inverse log
    """
    if type(fcast) in [pd.core.series.Series, pd.core.frame.DataFrame]:
        fcast = fcast.to_numpy()
    fcast = fcast.flatten()
    n_fcast = len(fcast)
    diff = scaler.inverse_transform(fcast)
    inv_diff = [np.log(actual.iloc[base_index].to_numpy())] # starting value
    for i in range(n_fcast-1):
        inv_diff.append(inv_diff[i] + np.log(actual).diff().iloc[base_index+i]) # get a hx1 list
    inversed = np.exp(inv_diff)
    return inversed.flatten()



def plot_fcast(df, method='hws'):
    """
    Plot forecast vs actual
    """
    fcast   = getattr(df, f'fcast_{method}')
    end_train = fcast.index[0]
    train   = df.loc[:end_train]
    test    = df.loc[end_train:].iloc[1:]
    
    fig, ax = plt.subplots(figsize=(6.5,3))
    ax.plot(train, color='black', alpha=0.4, lw=1)
    ax.plot(test, color='royalblue', lw=1)
    ax.plot(fcast.fcast1.shift(1), color='crimson', ls='dashed', lw=1)
    ax.set_xlim([datetime.date(2000, 1, 1), datetime.date(2019, 9, 1)])
    ax.set_ylim(0)
    sns.despine(bottom=False, left=True, ax=ax)