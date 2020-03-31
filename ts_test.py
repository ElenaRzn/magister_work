import numpy as np
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, output_file
from pandas import read_csv
from statsmodels.tsa.seasonal import seasonal_decompose


def get_figure(vacation):
    # # Read in vacation dataset
    # vacation = read_csv("df_vacation.csv", parse_dates=True)
    # print(vacation.head())
    # ------------------------------------------
    aapl = np.array(vacation['Num_Search_Vacation'])
    aapl_dates = np.array(vacation['Month'], dtype=np.datetime64)
    window_size = 50
    window = np.ones(window_size) / float(window_size)

    # output to static HTML file
    output_file("vacation.html", title="vacation.py example")

    # create a new plot with a a datetime axis type
    p = figure(width=800, height=350, x_axis_type="datetime")

    # add renderers
    p.line(aapl_dates, aapl, color='darkgrey')

    # NEW: customize by setting attributes
    p.title.text = "vacation"
    p.grid.grid_line_alpha = 0
    p.xaxis.axis_label = 'Date'
    p.yaxis.axis_label = 'Value'
    p.ygrid.band_fill_color = "olive"
    p.ygrid.band_fill_alpha = 0.1
    p.width_policy = 'max'
    p.height_policy = 'max'

    # show the results
    return p

# ------------------------------------------
from flask import Flask, render_template
from bokeh.embed import components
from bokeh.resources import INLINE

app = Flask(__name__)

@app.route('/index2')
def index():
    import_file_name = "df_vacation.csv"
    vacation = read_csv(import_file_name, parse_dates=True)
    # print(vacation.head())
    fig = get_figure(vacation)

    # grab the static resources
    js_resources = INLINE.render_js()
    css_resources = INLINE.render_css()

    # render template
    script, div = components(fig)
    html = render_template(
        'index.html',
        plot_script=script,
        plot_div=div,
        js_resources=js_resources,
        file_name=import_file_name
    )
    # return encode_utf8(html)
    return html

@app.route('/decompose2')
def decompose2():
    import_file_name = "df_vacation.csv"
    vacation = read_csv(import_file_name, parse_dates=True)
    ts1 = vacation['Num_Search_Vacation']
    # freq is the number of data points in a repeated cycle
    result = seasonal_decompose(ts1, model='additive', freq=12)
    # fig = get_multiple_figure(result)
    trend = get_single_figure(result.trend, "trend")
    seasonal = get_single_figure(result.seasonal, "seasonal")
    resid = get_single_figure(result.seasonal, "resid")
    observed = get_single_figure(result.seasonal, "observed")

    # grab the static resources
    js_resources = INLINE.render_js()

    # render template
    script1, div_trend = components(trend)
    script2, div_seasonal = components(seasonal)
    script3, div_resid = components(resid)
    script4, div_observed = components(observed)
    html = render_template(
        'decompose.html',
        plot_script1=script1,
        plot_script2=script2,
        plot_script3=script3,
        plot_script4=script4,
        plot_trend=div_trend,
        plot_seasonality=div_seasonal,
        plot_resid=div_resid,
        plot_observed=div_observed,
        js_resources=js_resources,
        file_name=import_file_name
    )
    # return encode_utf8(html)
    return html

def get_single_figure(data, text):
    window_size = 50

    # output to static HTML file
    output_file("vacation.html", title="vacation.py example")

    # create a new plot with a a datetime axis type
    p = figure(width=800, height=350, x_axis_type="datetime")

    # add renderers
    p.line(list(range(0, data.size)), data, color='darkgrey')

    # NEW: customize by setting attributes
    p.title.text = text
    p.grid.grid_line_alpha = 0
    p.xaxis.axis_label = 'Date'
    p.yaxis.axis_label = 'Value'
    p.ygrid.band_fill_color = "olive"
    p.ygrid.band_fill_alpha = 0.1
    p.width_policy = 'max'
    p.height_policy = 'max'

    # show the results
    return p

def get_multiple_figure(decompose):
    source = ColumnDataSource(data=dict(
        # maxSize = max(decompose.trend.size(), decompose.seasonal.size(), decompose.resid.size(), decompose.observed.size())
        x=list(range(0, decompose.trend.size)),
        y1=decompose.trend,
        y2=decompose.seasonal,
        y3=decompose.resid,
        y4=decompose.observed
    ))

    # output to static HTML file
    output_file("vacation.html", title="vacation.py example")

    range(0, 10)    # create a new plot with a a datetime axis type
    p = figure(width=800, height=350, x_axis_type="datetime")

    # add renderers
    p.vline_stack(['y1', 'y2', 'y3', 'y4'], x='x', source=source)
    # , color='darkgrey'

    # NEW: customize by setting attributes
    p.title.text = "vacation"
    p.grid.grid_line_alpha = 0
    p.xaxis.axis_label = 'Date'
    p.yaxis.axis_label = 'Value'
    p.ygrid.band_fill_color = "olive"
    p.ygrid.band_fill_alpha = 0.1
    p.width_policy = 'max'
    p.height_policy = 'max'

    # show the results
    return p

if __name__ == '__main__':
    app.run(debug=True)