# A module contain functions used for forecasting
import os
import numpy as np
from scipy import stats
import pandas as pd
import seaborn as sns
import datetime
import impyute as impy
import pmdarima as pm

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
from   matplotlib.ticker import FormatStrFormatter, ScalarFormatter
from numpy import random as npr

from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.stattools import grangercausalitytests as granger
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.api import VAR, VECM
from statsmodels.tsa.api import ExponentialSmoothing as ets
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.eval_measures import rmse, aic, bic
from statsmodels.tools.eval_measures import meanabs as mae
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras.backend as K

from types import SimpleNamespace
from itertools import product

import warnings
import joblib
warnings.filterwarnings("ignore")
pd.options.display.float_format = '{:,.4f}'.format

# import conversion dict
monthTH = pd.read_excel('../data/month.xlsx')
month_map = dict(zip(monthTH.short_th, monthTH.num))


class LogScaler:
    def transform(self, data):
        data = data if type(data) is np.ndarray else np.array(data)
        return np.log(data)    
    def inverse_transform(self, data):
        return np.exp(data)

class NoneScaler:
    def transform(self, data):
        data = data if type(data) is np.ndarray else np.array(data)
        return data    
    def inverse_transform(self, data):
        return data
    
# define functions

def mape(y, yhat):
    return np.abs((y-yhat)/y).mean(0)


def read_price(name):
    # y = actual data
    path = f'../data/{name}.xls'
    df = (pd.read_excel(path, sheet_name = 'price', skiprows = 2, na_values = '-')
          .melt(id_vars = 'ปี', var_name = 'month', value_name='y')
          .rename(columns = {'ปี':'year'})
          .assign(year = lambda x: x.year-543)
          .assign(month = lambda x: x.month.str.strip().map(month_map))
          .assign(y = lambda x: pd.to_numeric(x.y, errors='coerce'))
          .assign(date = lambda x: pd.to_datetime(
              '1' + '/' + x.month.astype(str) + '/' + x.year.astype(str),
              dayfirst = True,))
          .query('year<2020 & ~y.isna()')
          .set_index('date')
          .sort_index()
          .asfreq(freq='MS')
          .drop(columns=['year', 'month'])
          .interpolate(method='time')
       )
    return df

def test_stationary(df, name=None, col='y', transform_function = None, spec='c'):
    """
    Stationary test of a series
    Input: A dataframe with a specific column
    """
    if name == None:
        name = df.meta.shortname
    col_names = ['price', 'level', 'adf_stat', 'adf_pval', 'kpss_stat', 'kpss_pval']
    if transform_function is None:
        level = df[col]
    else:
        level = transform_function(df[col])
    diff1 = level.diff().dropna()

    result = []
    result.append([name, 'level', *adfuller(level, regression=spec)[:2], *kpss(level, regression=spec)[:2]])
    result.append([name, 'diff1', *adfuller(diff1)[:2], *kpss(diff1)[:2]])

    return pd.DataFrame(result, columns=col_names).set_index(['price', 'level']).round(4)


def data_transform(data, method=None):
    '''
    Tranform data and keep associated scaler
    
    Args:
        data (dataframe): a dataframe of interest
        method (str): a string indicating the transformation method default=None

    Return:
        a transformmed dataframe and a scaler
    '''
    
    if method=="standard":
        scaler = StandardScaler().fit(data)
    elif method=="minmax":
        scaler = MinMaxScaler().fit(data)
    elif method=="log":
        scaler = LogScaler()
    elif method=="box-cox":
        scaler = PowerTransformer(method='box-cox', standardize=True)
    else:
        scaler = NoneScaler()
        
    scaled = pd.DataFrame(index=data.index)
    scaled["y"] = scaler.transform(data)  
    
    return scaled, scaler

def data_difference(data, n_diff):
    correction = 0
    if n_diff > 0:
        correction = data.shift(n_diff)
    return data.diff(n_diff), correction


def inverse_difference(data, correction):
    return data + correction


def data_split(data, frac_validate=.2, frac_test=.2):
    """
    Split data into train, validatio, train2 and test set.

    Args:
        data (array or series): data of interest
        frac_valid (float): a fraction of data for validation set (default = 0.2)
        frac_test (float): a fraction of data for test set (default = 0.2)
    
    Returns:
        a tuple of 4 dataframe including train, train2, valid, and test.
    """
    
    n_size  = data.shape[0]
    n_test  = int(n_size*(frac_test))
    n_validate = int(n_size*(frac_validate))
    n_train = n_size - n_test - n_validate
    train  = data.iloc[:n_train].dropna()
    validate  = data.iloc[n_train:-n_test]
    train2 = data.iloc[:-n_test]
    test   = data.iloc[-n_test:]
    
    return train, validate, train2, test


def data_2keras(data, n_lag=12, n_forecast=4):
    """
    Rearrange data into keras format also keep time index

    Args:
        data (dataframe or series): dataframe or series with time index
        n_lag (int): number of lags to be used in the model
        n_fcast (int): number of steps to forecast
    
    Returns:
        x (dataframe): input for training
        y (dataframe): output for training
        date_idx: date index
    """

    df = pd.DataFrame(index=data.index)
    
    for i in reversed(range(n_lag+1)):
        df[f'x{i}'] = data.shift(i)
    
    for i in range(1, n_forecast+1):
        df[f'y{i}'] = data.shift(-i)
    df = df.dropna()
    date_idx = df.index
    x = df.iloc[:, :n_lag+1]
    y = df.iloc[:, n_lag+1:]
    
    return x, y, date_idx

def select_traintest(data, search_mode):
    if search_mode == True:
        train = getattr(data, "train")
        test = getattr(data, "validate")
    else:
        train = getattr(data, "train2")
        test = getattr(data, "test")
    return train, test


def model_configs(*args):
    configs = [arg for arg in args]
    return list(product(*configs))


def model_measure(data, yhat, config):
    
    config_name = ["-".join([str(i) for i in config])]
    y_ = []
    yhat_ = []
    n_yhat = yhat.shape[1]
    for i in range(n_yhat):
        y_.append(data.loc[yhat.index.shift(i)]['y'].to_numpy())
        yhat_.append(yhat[f'yhat{i+1}'].to_numpy())
        
    y_    = np.concatenate(y_, axis=0)
    yhat_ = np.concatenate(yhat_, axis=0)
    
    measures = [rmse, mae, mape]
    measure_names = ["rmse", "mae", "mape"]
    scores = (np.array([measure(y_, yhat_) for measure in measures])
              .reshape(1, len(measures))
             )
    scores = pd.DataFrame(scores, columns=measure_names, index=config_name)
    scores.index.name = 'config'
    return scores


def grid_search(data, model, n_forecast=4, n_repeat=1):
    model_fit, model_fcast, model_walk_forward, model_configs = model
    search_mode=True
    
    scores = []
    for config in model_configs:
        for i in range(n_repeat):
            try:
                yhat = model_walk_forward(data, config,
                                          search_mode=search_mode,
                                          n_forecast=n_forecast)
                score = model_measure(data, yhat, config)
                scores.append(score)
                print(score.round(4).to_dict('index'))
            except:
                pass
    res = pd.concat(scores).groupby(level='config', sort=False).mean()
    return res, model_configs[res.reset_index()['rmse'].idxmin()]


def forecast(data, model, config, n_forecast=4):
    model_fit, model_fcast, model_walk_forward, model_configs = model
    search_mode=False
    
    yhat = model_walk_forward(data, config,
                              search_mode=search_mode,
                              n_forecast=n_forecast)
    
    return yhat


def compute_error(data, model):
    train, validate, train2, test = data_split(data)
    yhat  = getattr(data, f'{model}_yhat')
    n_error = yhat.shape[1]
    error = pd.DataFrame()
    for i in range(n_error):
        error[f'err{i+1}'] = test.iloc[i:,0] - yhat.iloc[:,i].shift(i)
    return error


def compute_rmse(data, model):
    train, validate, train2, test = data_split(data)
    yhat  = getattr(data, f'{model}_yhat')
    n_test = len(yhat.iloc[:,0].dropna()) 
    n_error = yhat.shape[1]
    res = []
    for i in range(n_error):
        y_    = test.iloc[i:i+n_test].to_numpy().reshape(-1)
        yhat_ = yhat.iloc[:i+n_test,i].to_numpy().reshape(-1)
        res.append(rmse(y_, yhat_))
    return res


def plot_fcast(data, method='ets', step=1):
    """
    Plot forecast vs actual
    """
    
    train, validate, train2, test = data_split(data)
    yhat  = getattr(data, f'{method}_yhat')
    yhat = data.join(yhat)
    
    fig, ax = plt.subplots(figsize=(6.5,3))
    ax.plot(train2, color='black', alpha=0.4, lw=1)
    ax.plot(test, color='royalblue', lw=1)
    ax.plot(yhat[f'yhat{step}'].shift(step-1), color='crimson', ls='dashed', lw=1)
    
    ax.set_xlim([datetime.date(2008, 1, 1), datetime.date(2019, 9, 1)])
    ax.set_ylim(0)
    sns.despine(bottom=False, left=True, ax=ax)


