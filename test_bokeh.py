import bokeh.plotting  as plt
import numpy as np
# prepare some data
x = [1, 2, 3, 4, 5]
y = [6, 7, 2, 4, 5]

# output to static HTML file
plt.output_file("lines.html")

# create a new plot with a title and axis labels
p = plt.figure(title="simple line example", x_axis_label='x', y_axis_label='y', x_range=(0,1), y_range=(0,1))

# add a line renderer with legend and line thickness
# p.line(x, y, legend="Temp.", line_width=2)
p.image([np.random.random((4, 6))], 0, 0, 1, 1)

# show the results
plt.show(p)