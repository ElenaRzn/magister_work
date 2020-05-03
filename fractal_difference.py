#import all necessary libraries for analysis
import numpy as np
import pandas as pd
import quandl

#plotting libraries
from matplotlib import pyplot as plt

#Initialize figure size (in inches)
plt.rcParams['figure.figsize'] = [10,5]

pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

import os

#set API key (saved locally on machine)
quandl.ApiConfig.api_key = os.environ.get('quandl_api_key')

"""Computes fractionally differenced series
    Args:
        d: A float representing the differencing factor (any positive fractional)
        series: A pandas dataframe with one or more columns of time-series values to be differenced
        threshold: Threshold value past which we ignore weights
            (cutoff weight for window)
    Returns:
        diff_series: A numpy array of differenced series by d.
"""


# def fracDiff(series, d, threshold=1e-5):
def fracDiff(series, d, threshold=0):
    # compute weights using function above
    weights = findWeights_FFD(d, len(series), threshold)
    width = len(weights) - 1

    # forward fill through unavailable prices and create a temporary series to hold values
    curr_series = series.fillna(method='ffill').dropna()
    df_temp = pd.Series()
    i = 0;

    # for every value in the original series
    for i in range(0, len(curr_series) - 1):
        df_temp = df_temp._set_value(i, np.dot(weights[0:len(curr_series) - i].T, curr_series.loc[i:len(curr_series) - 1])[0])

    return df_temp


"""Computes the weights for our fractionally differenced features up to a given threshold
   requirement for fixed-window fractional differencing. 
    Args:
        d: A float representing the differencing factor
        length: An int length of series to be differenced
        threshold: A float representing the minimum threshold to include weights for
    Returns:
        A numpy array containing the weights to be applied to our time series
"""


def findWeights_FFD(d, length, threshold):
    # set first weight to be a 1 and k to be 1
    w, k = [1.], 1
    w_curr = 1

    # while we still have more weights to process, do the following:
    while (k < length):

        w_curr = (-w[-1] * (d - k + 1)) / k
        # if the current weight is below threshold, exit loop
        if (abs(w_curr) <= threshold):
            break
        # append coefficient to list if it passes above threshold condition
        w.append(w_curr)
        # increment k
        k += 1
    # make sure to convert it into a numpy array and reshape from a single row to a single
    # column so they can be applied to time-series values easier
    w = np.array(w[::-1]).reshape(-1, 1)

    return w