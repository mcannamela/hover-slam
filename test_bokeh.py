import bokeh.plotting  as plt
import numpy as np
import os

# output to static HTML file
plt.output_file(os.path.join("resources", "bokeh_tmp", "test.html"))

# create a new plot with a title and axis labels
p = plt.figure(title="higher powers command to paint the upper right corner black", x_axis_label='x', y_axis_label='y',
               x_range=(0, 1), y_range=(0, 1))

# add a line renderer with legend and line thickness
# p.line(x, y, legend="Temp.", line_width=2)
im = [
    [0, 0, 1, 0],
    [0, 0, .8, 0],
    [.2, .4, .6, .8]
]
p.image([im], 0, 0, 1, 1)

# show the results
plt.show(p)
