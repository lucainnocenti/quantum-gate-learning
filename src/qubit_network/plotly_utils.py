"""
Small collection of thin wrappers around plotly objects.
"""
import numbers


def hline(x0, x1, y, color='black', width=0.1, dash='solid'):
    # if `y` is a number a single line is returned
    if isinstance(y, numbers.Number):
        shape = dict(
            type='line',
            x0=x0, x1=x1, y0=y, y1=y,
            line=dict(color=color, width=width, dash=dash)
        )
        return shape
    # otherwise more than one line is returned
    else:
        shapes = [hline(x0, x1, y_, color=color, width=width, dash=dash)
                  for y_ in y]
        return shapes
