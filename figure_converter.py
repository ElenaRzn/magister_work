"""
Модуль для импорта данных из .csv файлов.
"""
import numpy as np
from bokeh.io import output_file
from bokeh.models import ColumnDataSource, Legend, LegendItem
from bokeh.plotting import figure
from pandas import read_csv

'''
 Чтение из .csv файла.
 import_file_name - 
'''


def load_csv(import_file_name, info_column, date_column):
    time_series = read_csv(import_file_name, parse_dates=True)
    fig = get_figure(np.array(time_series[info_column]), np.array(time_series[date_column], dtype=np.datetime64))
    return fig, time_series


def get_figure(infos, dates):
    window_size = 50
    window = np.ones(window_size) / float(window_size)

    # output to static HTML file
    output_file("display.html", title="Time Series")

    # create a new plot with a a datetime axis type
    p = figure(width=800, height=350, x_axis_type="datetime")

    # add renderers
    p.line(dates, infos, color='darkgrey')

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


def get_multi_figure(x1, x2, x3, y1, y2, y3, text):
    window_size = 50

    # output to static HTML file
    output_file("multi.html", title="vacation.py example")

    # create a new plot with a a datetime axis type
    p = figure(width=800, height=350)

    # add renderers
    # source = ColumnDataSource(data=dict(
    #     x=dates,
    #     y1=infos,
    #     y2=model,
    # ))

    # p.vline_stack(x='x', y=['y1', 'y2'], source=source, color=["firebrick", "navy"])
    r = p.multi_line(xs=[x1, x2, x3], ys=[y1, y2, y3],
                 color=['blue', 'green', 'red'])
    legend = Legend(items=[
        LegendItem(label="test", renderers=[r], index=0),
        LegendItem(label="actual", renderers=[r], index=1),
        LegendItem(label="predicted", renderers=[r], index=2),
    ])
    p.add_layout(legend)

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


def get_multi_figure(x1, x2, y1, y2, text):
    window_size = 50

    # output to static HTML file
    output_file("multi.html", title="vacation.py example")

    # create a new plot with a a datetime axis type
    p = figure(width=800, height=350)

    # add renderers
    # source = ColumnDataSource(data=dict(
    #     x=dates,
    #     y1=infos,
    #     y2=model,
    # ))

    # p.vline_stack(x='x', y=['y1', 'y2'], source=source, color=["firebrick", "navy"])
    r = p.multi_line(xs=[x1, x2], ys=[y1, y2],
                 color=['blue', 'green'])
    legend = Legend(items=[
        LegendItem(label="model", renderers=[r], index=0),
        LegendItem(label="actual", renderers=[r], index=1),
    ])
    p.add_layout(legend)

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


def get_single_chart(data, text):
    # output to static HTML file
    output_file("vacation.html", title="vacation.py example")

    # create a new plot with a a datetime axis type
    p = figure(width=150, height=250)

    # add renderers
    p.vbar(x=list(range(0, data.size)), width=0.05, bottom=0, top=data, color="firebrick")

    # NEW: customize by setting attributes
    p.title.text = text
    p.grid.grid_line_alpha = 0
    p.xaxis.axis_label = 'Lags'
    p.yaxis.axis_label = 'Value'
    p.ygrid.band_fill_color = "olive"
    p.ygrid.band_fill_alpha = 0.1
    p.width_policy = 'max'
    p.height_policy = 'max'

    # show the results
    return p