from bokeh.plotting import figure, output_file, save
output_file('data_science_popularity.html')


p = figure(title='data science', x_axis_label='Mes', y_axis_label='data science')
p.line(df['Mes'], df['data science'], legend='popularity', line_width=2)
save(p)