import statsmodels
from bokeh.embed import components

from figure_converter import get_single_chart


def get_acf(time_series_data):
    plt_acf = statsmodels.tsa.stattools.acf(time_series_data, nlags=12)
    acf = get_single_chart(plt_acf, "ACF")
    js_acf, div_acf = components(acf)
    return js_acf, div_acf


def get_pacf(time_series_data):
    # Plot Partial autocorrelation function (PACF)
    plt_pacf = statsmodels.tsa.stattools.pacf(time_series_data, nlags=12)
    pacf = get_single_chart(plt_pacf, "PACF")
    js_pacf, div_pacf = components(pacf)
    return js_pacf, div_pacf