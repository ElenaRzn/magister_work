import numpy as np
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, output_file, Figure
from pandas import read_csv
from statsmodels.tsa.seasonal import seasonal_decompose

from flask import Flask, render_template, redirect, url_for, request
from bokeh.embed import components
from bokeh.resources import INLINE

from load import load_csv

app = Flask(__name__)

fig = []

@app.route('/', methods=['GET'])
def home():
    return render_template("home.html")


@app.route('/index', methods=['GET'])
def index():
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
    fig.append(load_csv(request.form['file'], request.form['data'], request.form['date']))
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)