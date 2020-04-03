import numpy as np
import pandas as pd
from bokeh.embed import components
from bokeh.resources import INLINE
from flask import Flask, render_template, redirect, url_for, request
from pandas.io.parsers import read_csv
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARMA

from autocorrelation import get_acf, get_pacf
from figure_converter import get_figure, get_single_figure, get_multi_figure

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


@app.route('/stationary', defaults={'integer_diff': None, 'log_diff':None}, methods=['GET'])
@app.route('/stationary/<integer_diff>', defaults={'log_diff': None}, methods=['GET'])
@app.route('/stationary/log_diff', defaults={'integer_diff': None}, methods=['GET'])
def stationary(integer_diff, log_diff):
    if time_series is None:
        return redirect(url_for('home'))

    to_analyse = []
    if integer_diff is not None:
        to_analyse = time_series['integer_diff']
    elif log_diff is not None:
        to_analyse = time_series['log_diff']
    else:
        to_analyse = time_series[information_column]

    time_series['result'] = to_analyse

    ts_result = adfuller(to_analyse)
    coefficient = to_analyse.autocorr()

    js_acf, div_acf = get_acf(to_analyse)
    js_pacf, div_pacf = get_pacf(to_analyse)

    js_resources = INLINE.render_js()

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
    return redirect(url_for('stationary', integer_diff=True))


@app.route('/log', methods=['POST'])
def log():
    global time_series
    time_series["log_diff"] = np.log(time_series[information_column])
    time_series.dropna(inplace=True)
    return redirect(url_for('stationary', log_deff=True))

@app.route('/accept', methods=['POST'])
def accept():
    global time_series
    time_series[information_column] = time_series['result']
    return redirect(url_for('index'))


@app.route('/model', methods=['GET'])
def model():
    if time_series is None:
        return redirect(url_for('home'))
    js_acf, div_acf = get_acf(time_series[information_column])
    js_pacf, div_pacf = get_pacf(time_series[information_column])
    js_resources = INLINE.render_js()
    html = render_template(
        'model.html',
        js_resources=js_resources,
        plot_acf=div_acf,
        plot_pacf=div_pacf,
        script_acf=js_acf,
        script_pacf=js_pacf
    )
    return html


@app.route('/arma', methods=['POST'])
def arma():
    p = int(request.form['ar'])
    d = int(request.form['ma'])

    # Create Training and Test
    train = time_series[information_column][:len(time_series[information_column]) - 15]
    test = time_series[information_column][len(time_series[information_column]) - 15:]

    mod = ARMA(train, order=(p, d))

    res = mod.fit()

    # Print out summary information on the fit
    summary = res.summary()

    # Print out the estimate for the constant and for theta
    params = res.params

    # prediction = res.predict(start=len(time_series[information_column]), end=len(time_series[information_column]) + 10)

    forecasts, stderr, conf_int = res.forecast(15, alpha=0.05)
    # figure = get_multi_figure(time_series[information_column], time_series[date_column], prediction, 'ARIMA')
    figure = get_multi_figure(list(range(0, train.size)), list(range(train.size, time_series[information_column].size)),
                              list(range(train.size, time_series[information_column].size)),
                              train, test, forecasts, 'ARMA')
    script, div = components(figure)

    js_resources = INLINE.render_js()
    html = render_template(
        'result.html',
        js_resources=js_resources,
        plot_model=div,
        plot_script=script,
        summary = summary,
        params = params
    )
    return html


# @app.route('/arma', methods=['POST'])
# def arma():
#     p = int(request.form['ar'])
#     d = int(request.form['ma'])
#
#     mod = ARMA(time_series[information_column], order=(int(request.form['ar']), int(request.form['ma'])))
#
#     res = mod.fit()
#
#     # Print out summary information on the fit
#     summary = res.summary()
#
#     # Print out the estimate for the constant and for theta
#     params = res.params
#
#     prediction = res.predict(start=len(time_series[information_column]), end=2025)
#     # figure = get_multi_figure(time_series[information_column], time_series[date_column], prediction, 'ARIMA')
#     figure = get_figure(time_series[date_column], time_series[information_column])
#     script, div = components(figure)
#
#     js_resources = INLINE.render_js()
#     html = render_template(
#         'result.html',
#         js_resources=js_resources,
#         plot_model=div,
#         plot_script=script,
#         summary = summary,
#         params = params
#     )
#     return html


if __name__ == '__main__':
    app.run(debug=True)