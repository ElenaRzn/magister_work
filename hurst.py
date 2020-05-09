import numpy as np
import pandas as pd


def hurst(X):
    """ Compute the Hurst exponent of X. If the output H=0.5,the behavior
    of the time-series is similar to random walk. If H<0.5, the time-series
    cover less "distance" than a random walk, vice verse.
    Parameters
    ----------
    X
        list
        a time series
    Returns
    -------
    H
        float
        Hurst exponent
    """
    N = X.size
    T = np.arange(1, N + 1)
    Y = np.cumsum(X)
    Ave_T = Y / T

    S_T = np.zeros(N)
    R_T = np.zeros(N)

    for i in range(N):
        S_T[i] = np.std(X[:i + 1])
        X_T = Y - T * Ave_T[i]
        R_T[i] = np.ptp(X_T[:i + 1])

    R_S = R_T / S_T
    R_S = np.log(R_S)[1:]
    R_S = R_S[~np.isnan(R_S)]
    n = np.log(T)[2:]
    A = np.column_stack((n, np.ones(n.size)))
    if A.size != R_S.size * 2 :
        R_S = R_S[1:]
    [m, c] = np.linalg.lstsq(A, R_S)[0]
    H = m
    return H