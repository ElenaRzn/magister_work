from pandas import read_csv
import numpy as np
from bokeh.plotting import figure, output_file, show

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
    p.yaxis.axis_label = 'Price'
    p.ygrid.band_fill_color = "olive"
    p.ygrid.band_fill_alpha = 0.1

    # show the results
    return p

# ------------------------------------------
from flask import Flask, render_template
from bokeh.embed import components
from bokeh.resources import INLINE
from bokeh.util.string import encode_utf8

app = Flask(__name__)

@app.route('/')
def index():
    vacation = read_csv("df_vacation.csv", parse_dates=True)
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
        js_resources=js_resources
    )
    # return encode_utf8(html)
    return html

@app.route('/bokeh')
def bokeh():
    vacation = read_csv("df_vacation.csv", parse_dates=True)
    print(vacation.head())
    fig = get_figure(vacation)

    # grab the static resources
    js_resources = INLINE.render_js()
    css_resources = INLINE.render_css()

    # render template
    script, div = components(fig)
    # html = render_template(
    #     'index.html',
    #     plot_script=script,
    #     plot_div=div,
    #     js_resources=js_resources,
    #     css_resources=css_resources
    # )
    # return encode_utf8(html)
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)