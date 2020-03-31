"""
Модуль для импорта данных из .csv файлов.
"""
import numpy as np
from bokeh.io import output_file
from bokeh.plotting import figure
from pandas import read_csv

'''
 Чтение из .csv файла.
 import_file_name - 
'''


def load_csv(import_file_name, info_column, date_column):
    time_series = read_csv(import_file_name, parse_dates=True)
    fig = get_figure(np.array(time_series[info_column]), np.array(time_series[date_column]))
    return fig


def get_figure(aapl, aapl_dates):
    window_size = 50
    window = np.ones(window_size) / float(window_size)

    # output to static HTML file
    output_file("display.html", title="Time Series")

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