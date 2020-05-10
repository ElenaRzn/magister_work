"""
Python code for fractional differencing of pandas time series
illustrating the concepts of the article "Preserving Memory in Stationary Time Series"
by Simon Kuttruf
https://towardsdatascience.com/preserving-memory-in-stationary-time-series-6842f7581800

While this code is dedicated to the public domain for use without permission, the author disclaims any liability in connection with the use of this code.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def getWeights(d, lags):
    # return the weights from the series expansion of the differencing operator
    # for real orders d and up to lags coefficients
    w = [1]
    for k in range(1, lags):
        w.append(-w[-1] * ((d - k + 1)) / k)
    w = np.array(w).reshape(-1, 1)
    return w


def ts_differencing(series, order, lag_cutoff):
    # return the time series resulting from (fractional) differencing
    # for real orders order up to lag_cutoff coefficients


    weights = getWeights(order, lag_cutoff)
    res = 0
    for k in range(lag_cutoff):
        res += weights[k] * series.shift(k).fillna(0)
    return res

from statsmodels.tsa.stattools import adfuller


def memoryVsCorr(series, dRange, numberPlots, lag_cutoff, seriesName):
    # return a data frame and plot comparing adf statistics and linear correlation
    # for numberPlots orders of differencing in the interval dRange up to a lag_cutoff coefficients

    interval = np.linspace(dRange[0], dRange[1], numberPlots)
    result = pd.DataFrame(np.zeros((len(interval), 4)))
    result.columns = ['order', 'adf', 'corr', '5%']
    result['order'] = interval
    for counter, order in enumerate(interval):
        seq_traf = ts_differencing(series, order, lag_cutoff)
        res = adfuller(seq_traf, maxlag=1, regression='c')  # autolag='AIC'
        result.loc[counter, 'adf'] = res[0]
        result.loc[counter, '5%'] = res[4]['5%']
        result.loc[counter, 'corr'] = np.corrcoef(series[lag_cutoff:].fillna(0), seq_traf)[0, 1]
    # plotMemoryVsCorr(result, seriesName)
    return result