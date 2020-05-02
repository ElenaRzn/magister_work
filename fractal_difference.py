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
def fracDiff(series, d, threshold=1e-3):
    # compute weights using function above
    weights = findWeights_FFD(d, len(series), threshold)
    width = len(weights) - 1

    df = []
    # for each series to be differenced, apply weights to appropriate prices and save
    for name in series.columns:
        if name == 'Month' :
            continue

        # forward fill through unavailable prices and create a temporary series to hold values
        curr_series = series[[name]].fillna(method='ffill').dropna()
        df_temp = pd.Series()
        i = 0;

        # loop through all values that fall into range to be fractionally differenced
        for iloc1 in range(width, curr_series.shape[0]):

            # set values for first and last time-series point to be used in current pass of fractional
            # difference
            loc0 = curr_series.index[iloc1 - width]
            loc1 = curr_series.index[iloc1]

            # make sure current value is valid
            if not np.isfinite(curr_series.loc[loc1, name]):
                continue

            # dot product of weights with values from first and last indices
            # df_temp[loc1] = np.dot(weights.T, curr_series.loc[loc0:loc1])[0, 0]
            df_temp = df_temp._set_value(i, np.dot(weights.T, curr_series.loc[loc0:loc1])[0, 0])
            i = i + 1
            # df_temp = np.npdot(weights.T, curr_series.loc[loc0:loc1])[0, 0]

        # df[name] = df_temp.copy(deep=True)
        # df.add(df_temp.copy())
    # df = pd.concat(df, axis=1)

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