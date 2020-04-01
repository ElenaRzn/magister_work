import numpy as np
from bokeh.embed import components
from bokeh.resources import INLINE
from flask import Flask, render_template, redirect, url_for, request
from pandas.io.parsers import read_csv
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
# from statsmodels.graphics.tsaplots import plot_acf
# from statsmodels.graphics.tsaplots import plot_pacf
import statsmodels.tsa.stattools

from figure_converter import get_figure, get_single_figure, get_single_chart

app = Flask(__name__)

fig = []
# time_series: Union[Union[TextFileReader, Series, DataFrame, None], Any]
time_series = None
information_column = ""
date_column = ""



@app.route('/', methods=['GET'])
def home():
    return render_template("home.html")


@app.route('/index', methods=['GET'])
def index():
    if time_series is None:
        return redirect(url_for('home'))
    # grab the static resources
    js_resources = INLINE.render_js()

    # render template
    script, div = components(fig[0])
    html = render_template(
        'index.html',
        plot_script=script,
        plot_div=div,
        js_resources=js_resources,
        file_name='test'
    )
    # return encode_utf8(html)
    return html


@app.route('/load', methods=['POST'])
def load():
    global time_series
    time_series = read_csv(request.form['file'], parse_dates=True)
    global information_column
    information_column = request.form['data']
    global date_column
    date_column = request.form['date']
    global fig
    fig.append(get_figure(np.array(time_series[information_column]), np.array(time_series[date_column], dtype=np.datetime64)))
    return redirect(url_for('index'))


@app.route('/decompose', methods=['GET'])
def decompose():
    if time_series is None:
        return redirect(url_for('home'))
    result = seasonal_decompose(time_series[information_column], model='additive', freq=12)
    # fig = get_multiple_figure(result)
    trend = get_single_figure(result.trend, "trend")
    seasonal = get_single_figure(result.seasonal, "seasonal")
    resid = get_single_figure(result.resid, "resid")
    observed = get_single_figure(result.observed, "observed")

    # grab the static resources
    js_resources = INLINE.render_js()

    # render template
    js_trend, div_trend = components(trend)
    js_seasonal, div_seasonal = components(seasonal)
    js_resid, div_resid = components(resid)
    js_observed, div_observed = components(observed)
    html = render_template(
        'decompose.html',
        script_trend=js_trend,
        script_seasonal=js_seasonal,
        script_resid=js_resid,
        script_observed=js_observed,
        plot_trend=div_trend,
        plot_seasonality=div_seasonal,
        plot_resid=div_resid,
        plot_observed=div_observed,
        js_resources=js_resources,
        file_name="test"
    )
    return html


@app.route('/stationary', methods=['GET'])
def stationary():
    if time_series is None:
        return redirect(url_for('home'))

    # Dickey-Fuller Test
    ts_result = adfuller(time_series[information_column])
    # Autocorrelation
    coefficient = time_series[information_column].autocorr()

    # Plot ACF to visualize the autocorrelation

    plt_acf = statsmodels.tsa.stattools.acf(time_series[information_column], nlags=12)
    acf = get_single_chart(plt_acf, "ACF")
    js_acf, div_acf = components(acf)

    # Plot Partial autocorrelation function (PACF)
    plt_pacf = statsmodels.tsa.stattools.pacf(time_series[information_column], nlags=12)
    pacf = get_single_chart(plt_pacf, "PACF")
    js_pacf, div_pacf = components(pacf)

    # grab the static resources
    js_resources = INLINE.render_js()

    if 'integer_diff' in time_series.columns:
        # Dickey-Fuller Test
        ts_result_integer = adfuller(time_series["integer_diff"])
        # Autocorrelation
        coefficient_integer = time_series["integer_diff"].autocorr()

        # Plot ACF to visualize the autocorrelation

        plt_acf_integer = statsmodels.tsa.stattools.acf(time_series["integer_diff"], nlags=12)
        acf_integer = get_single_chart(plt_acf_integer, "ACF")
        js_acf_integer, div_acf_integer = components(acf_integer)

        # Plot Partial autocorrelation function (PACF)
        plt_pacf_integer = statsmodels.tsa.stattools.pacf(time_series["integer_diff"], nlags=12)
        pacf_integer = get_single_chart(plt_pacf_integer, "PACF")
        js_pacf_integer, div_pacf_integer = components(pacf_integer)

        # grab the static resources
        html = render_template(
            'stationary.html',
            js_resources=js_resources,
            dft_statistic_result=ts_result[0],
            dft_p_result=ts_result[1],
            dft_critical_result=ts_result[4],
            autocorrelation=coefficient,
            plot_acf=div_acf,
            plot_pacf=div_pacf,
            script_acf=js_acf,
            script_pacf=js_pacf,
            dft_statistic_result_integer=ts_result_integer[0],
            dft_p_result_integer=ts_result_integer[1],
            dft_critical_result_integer=ts_result_integer[4],
            autocorrelation_integer=coefficient_integer,
            plot_acf_integer=div_acf_integer,
            plot_pacf_integer=div_pacf_integer,
            script_acf_integer=js_acf_integer,
            script_pacf_integer=js_pacf_integer,
            integer_result=True

        )
        return html

    html = render_template(
        'stationary.html',
        js_resources=js_resources,
        dft_statistic_result=ts_result[0],
        dft_p_result=ts_result[1],
        dft_critical_result=ts_result[4],
        autocorrelation = coefficient,
        plot_acf=div_acf,
        plot_pacf=div_pacf,
        script_acf=js_acf,
        script_pacf=js_pacf
    )
    return html



@app.route('/integer', methods=['POST'])
def integer():
    global time_series
    # Example of second differencing
    time_series["integer_diff"] = time_series[information_column].diff()
    time_series.dropna(inplace=True)
    return redirect(url_for('stationary', integer_result=True))

# @app.route('/log', methods=['POST'])
# def integer():
#     global time_series
#     # Example of second differencing
#     time_series["log_diff"] = time_series[information_column].log()
#     time_series.dropna(inplace=True)
#     return redirect(url_for('stationary'))


if __name__ == '__main__':
    app.run(debug=True)