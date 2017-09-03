"""This file contains code for use with "Think Stats".

By Allen B. Downey, available from greenteapress.com

Copyright 2014 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

from __future__ import print_function

import logging
import math
import matplotlib
import matplotlib.pyplot as pyplot
import numpy as np
import pandas

# customize some matplotlib attributes
# matplotlib.rc('figure', figsize=(4, 3))

# matplotlib.rc('font', size=14.0)
# matplotlib.rc('axes', labelsize=22.0, titlesize=22.0)
# matplotlib.rc('legend', fontsize=20.0)

# matplotlib.rc('xtick.major', size=6.0)
# matplotlib.rc('xtick.minor', size=3.0)

# matplotlib.rc('ytick.major', size=6.0)
# matplotlib.rc('ytick.minor', size=3.0)


class _Brewer(object):
    """Encapsulate a nice sequence of colors.

    Shades of blue that look good in color and can be distinguished
    in grayscale (up to a point).

    Borrowed from http://colorbrewer2.org/
    """

    color_iter = None

    colors = ['#081D58',
              '#253494',
              '#225EA8',
              '#1D91C0',
              '#41B6C4',
              '#7FCDBB',
              '#C7E9B4',
              '#EDF8B1',
              '#FFFFD9']

    # lists that indicate which colors to use depending on how many are used
    which_colors = [[],
                    [1],
                    [1, 3],
                    [0, 2, 4],
                    [0, 2, 4, 6],
                    [0, 2, 3, 5, 6],
                    [0, 2, 3, 4, 5, 6],
                    [0, 1, 2, 3, 4, 5, 6],
                    ]

    @classmethod
    def get_colors(cls):
        """Return the list of colors."""
        return cls.colors

    @classmethod
    def color_generator(cls, n):
        """Return an iterator of color strings.

        n: how many colors will be used
        """
        for i in cls.which_colors[n]:
            yield cls.colors[i]
        raise StopIteration('Ran out of colors in _Brewer.color_generator')

    @classmethod
    def initialize_iter(cls, num):
        """Initialize the color iterator with the given number of colors."""
        cls.color_iter = cls.color_generator(num)

    @classmethod
    def clear_iter(cls):
        """Set the color iterator to None."""
        cls.color_iter = None

    @classmethod
    def get_iter(cls):
        """Get the color iterator."""
        if cls.color_iter is None:
            cls.initialize_iter(7)

        return cls.color_iter


def pre_plot(num=None, rows=None, cols=None):
    """Take hints about what's coming.

    num: number of lines that will be plotted
    rows: number of rows of subplots
    cols: number of columns of subplots
    """
    if num:
        _Brewer.initialize_iter(num)

    if rows is None and cols is None:
        return

    if rows is not None and cols is None:
        cols = 1

    if cols is not None and rows is None:
        rows = 1

    # resize the image, depending on the number of rows and cols
    size_map = {(1, 1): (8, 6),
                (1, 2): (14, 6),
                (1, 3): (14, 6),
                (2, 2): (10, 10),
                (2, 3): (16, 10),
                (3, 1): (8, 10),
                }

    if (rows, cols) in size_map:
        fig = pyplot.gcf()
        fig.set_size_inches(*size_map[rows, cols])

    # create the first subplot
    if rows > 1 or cols > 1:
        pyplot.subplot(rows, cols, 1)
        global SUBPLOT_ROWS, SUBPLOT_COLS
        SUBPLOT_ROWS = rows
        SUBPLOT_COLS = cols


def sub_plot(plot_number, rows=None, cols=None):
    """Configure the number of subplots and changes the current plot.

    rows: int
    cols: int
    plot_number: int
    """
    rows = rows or SUBPLOT_ROWS
    cols = cols or SUBPLOT_COLS
    pyplot.subplot(rows, cols, plot_number)


def _underride(d, **options):
    """Add key-value pairs to d only if key is not in d.

    If d is None, create a new dictionary.

    d: dictionary
    options: keyword args to add to d
    """
    if d is None:
        d = {}

    for key, val in options.items():
        d.setdefault(key, val)

    return d


def clear_figure():
    """Clear the figure and any hints that have been set."""
    _Brewer.clear_iter()
    pyplot.clf()
    fig = pyplot.gcf()
    fig.set_size_inches(8, 6)


def figure(**options):
    """Set options for the current figure."""
    _underride(options, figsize=(6, 8))
    pyplot.figure(**options)


def _underride_color(options):
    if 'color' in options:
        return options

    color_iter = _Brewer.get_iter()

    if color_iter:
        try:
            options['color'] = next(color_iter)
        except StopIteration:
            print('Warning: Brewer ran out of colors.')
            _Brewer.ClearIter()
    return options


def plot(obj, ys=None, style='', **options):
    """Plot a line.

    Args:
      obj: sequence of x values, or Series, or anything with render()
      ys: sequence of y values
      style: style string passed along to pyplot.plot
      options: keyword args passed to pyplot.plot
    """
    options = _underride_color(options)
    label = getattr(obj, 'name', '_nolegend_')
    options = _underride(options, linewidth=3, alpha=0.8, label=label)

    xs = obj
    if ys is None:
        if hasattr(obj, 'render'):
            xs, ys = obj.render()
        if isinstance(obj, pandas.Series):
            ys = obj.values
            xs = obj.index

    if ys is None:
        pyplot.plot(xs, style, **options)
    else:
        pyplot.plot(xs, ys, style, **options)


def fill_between(xs, y1, y2=None, where=None, **options):
    """Plot a line.

    Args:
      xs: sequence of x values
      y1: sequence of y values
      y2: sequence of y values
      where: sequence of boolean
      options: keyword args passed to pyplot.fill_between
    """
    options = _underride_color(options)
    options = _underride(options, linewidth=0, alpha=0.5)
    pyplot.fill_between(xs, y1, y2, where, **options)


def bar(xs, ys, **options):
    """Plot a line.

    Args:
      xs: sequence of x values
      ys: sequence of y values
      options: keyword args passed to pyplot.bar
    """
    options = _underride_color(options)
    options = _underride(options, linewidth=0, alpha=0.6)
    pyplot.bar(xs, ys, **options)


def scatter(xs, ys=None, **options):
    """Make a scatter plot.

    xs: x values
    ys: y values
    options: options passed to pyplot.scatter
    """
    options = _underride(options, color='blue', alpha=0.2, s=30,
                         edgecolors='none')

    if ys is None and isinstance(xs, pandas.Series):
        ys = xs.values
        xs = xs.index

    pyplot.scatter(xs, ys, **options)


def hex_bin(xs, ys, **options):
    """Make a scatter plot.

    xs: x values
    ys: y values
    options: options passed to pyplot.scatter
    """
    options = _underride(options, cmap=matplotlib.cm.Blues)
    pyplot.hexbin(xs, ys, **options)


def pdf(pdf, **options):
    """Plot a Pdf, Pmf, or Hist as a line.

    Args:
      pdf: Pdf, Pmf, or Hist object
      options: keyword args passed to pyplot.plot
    """
    low, high = options.pop('low', None), options.pop('high', None)
    n = options.pop('n', 101)
    xs, ps = pdf.render(low=low, high=high, n=n)
    options = _underride(options, label=pdf.name)
    plot(xs, ps, **options)


def pdfs(pdfs, **options):
    """Plot a sequence of PDFs.

    Options are passed along for all PDFs.  If you want different
    options for each pdf, make multiple calls to Pdf.

    Args:
      pdfs: sequence of PDF objects
      options: keyword args passed to pyplot.plot
    """
    for pdf in pdfs:
        pdf(pdf, **options)


def hist(hist, **options):
    """Plot a Pmf or Hist with a bar plot.

    The default width of the bars is based on the minimum difference
    between values in the Hist.  If that's too small, you can override
    it by providing a width keyword argument, in the same units
    as the values.

    Args:
      hist: Hist or Pmf object
      options: keyword args passed to pyplot.bar
    """
    # find the minimum distance between adjacent values
    xs, ys = hist.render()

    if 'width' not in options:
        try:
            options['width'] = 0.9 * np.diff(xs).min()
        except TypeError:
            logging.warning("Hist: Can't compute bar width automatically."
                            "Check for non-numeric types in Hist."
                            "Or try providing width option."
                            )

    options = _underride(options, label=hist.name)
    options = _underride(options, align='center')
    if options['align'] == 'left':
        options['align'] = 'edge'
    elif options['align'] == 'right':
        options['align'] = 'edge'
        options['width'] *= -1

    bar(xs, ys, **options)


def hists(hists, **options):
    """Plot two histograms as interleaved bar plots.

    Options are passed along for all PMFs.  If you want different
    options for each pmf, make multiple calls to Pmf.

    Args:
      hists: list of two Hist or Pmf objects
      options: keyword args passed to pyplot.plot
    """
    for hist in hists:
        hist(hist, **options)


def pmf(pmf, **options):
    """Plot a Pmf or Hist as a line.

    Args:
      pmf: Hist or Pmf object
      options: keyword args passed to pyplot.plot
    """
    xs, ys = pmf.render()
    low, high = min(xs), max(xs)

    width = options.pop('width', None)
    if width is None:
        try:
            width = np.diff(xs).min()
        except TypeError:
            logging.warning("Pmf: Can't compute bar width automatically."
                            "Check for non-numeric types in Pmf."
                            "Or try providing width option.")
    points = []

    lastx = np.nan
    lasty = 0
    for x, y in zip(xs, ys):
        if (x - lastx) > 1e-5:
            points.append((lastx, 0))
            points.append((x, 0))

        points.append((x, lasty))
        points.append((x, y))
        points.append((x + width, y))

        lastx = x + width
        lasty = y
    points.append((lastx, 0))
    pxs, pys = zip(*points)

    align = options.pop('align', 'center')
    if align == 'center':
        pxs = np.array(pxs) - width / 2.0
    if align == 'right':
        pxs = np.array(pxs) - width

    options = _underride(options, label=pmf.name)
    plot(pxs, pys, **options)


def pmfs(pmfs, **options):
    """Plot a sequence of PMFs.

    Options are passed along for all PMFs.  If you want different
    options for each pmf, make multiple calls to Pmf.

    Args:
      pmfs: sequence of PMF objects
      options: keyword args passed to pyplot.plot
    """
    for apmf in pmfs:
        pmf(apmf, **options)


def diff(t):
    """Compute the differences between adjacent elements in a sequence.

    Args:
        t: sequence of number

    Returns:
        sequence of differences (length one less than t)
    """
    diffs = [t[i + 1] - t[i] for i in range(len(t) - 1)]
    return diffs


def cdf(cdf, complement=False, transform=None, **options):
    """Plot a CDF as a line.

    Args:
      cdf: Cdf object
      complement: boolean, whether to plot the complementary CDF
      transform: string, one of 'exponential', 'pareto', 'weibull', 'gumbel'
      options: keyword args passed to pyplot.plot

    Returns:
      dictionary with the scale options that should be passed to
      Config, Show or Save.
    """
    xs, ps = cdf.render()
    xs = np.asarray(xs)
    ps = np.asarray(ps)

    scale = dict(xscale='linear', yscale='linear')

    for s in ['xscale', 'yscale']:
        if s in options:
            scale[s] = options.pop(s)

    if transform == 'exponential':
        complement = True
        scale['yscale'] = 'log'

    if transform == 'pareto':
        complement = True
        scale['yscale'] = 'log'
        scale['xscale'] = 'log'

    if complement:
        ps = [1.0 - p for p in ps]

    if transform == 'weibull':
        xs = np.delete(xs, -1)
        ps = np.delete(ps, -1)
        ps = [-math.log(1.0 - p) for p in ps]
        scale['xscale'] = 'log'
        scale['yscale'] = 'log'

    if transform == 'gumbel':
        xs = xp.delete(xs, 0)
        ps = np.delete(ps, 0)
        ps = [-math.log(p) for p in ps]
        scale['yscale'] = 'log'

    options = _underride(options, label=cdf.name)
    plot(xs, ps, **options)
    return scale


def cdfs(cdfs, complement=False, transform=None, **options):
    """Plot a sequence of CDFs.

    cdfs: sequence of CDF objects
    complement: boolean, whether to plot the complementary CDF
    transform: string, one of 'exponential', 'pareto', 'weibull', 'gumbel'
    options: keyword args passed to pyplot.plot
    """
    for cdf in cdfs:
        cdf(cdf, complement, transform, **options)


def contour(obj, pcolor=False, contour=True, imshow=False, **options):
    """Make a contour plot.

    d: map from (x, y) to z, or object that provides GetDict
    pcolor: boolean, whether to make a pseudocolor plot
    contour: boolean, whether to make a contour plot
    imshow: boolean, whether to use pyplot.imshow
    options: keyword args passed to pyplot.pcolor and/or pyplot.contour
    """
    try:
        d = obj.GetDict()
    except AttributeError:
        d = obj

    _underride(options, linewidth=3, cmap=matplotlib.cm.Blues)

    xs, ys = zip(*d.keys())
    xs = sorted(set(xs))
    ys = sorted(set(ys))

    x, y = np.meshgrid(xs, ys)

    def func(x, y):
        return d.get((x, y), 0)

    func = np.vectorize(func)
    z = func(x, y)

    x_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
    axes = pyplot.gca()
    axes.xaxis.set_major_formatter(x_formatter)

    if pcolor:
        pyplot.pcolormesh(x, y, z, **options)
    if contour:
        cs = pyplot.contour(x, y, z, **options)
        pyplot.clabel(cs, inline=1, fontsize=10)
    if imshow:
        extent = xs[0], xs[-1], ys[0], ys[-1]
        pyplot.imshow(z, extent=extent, **options)


def pseudo_color(xs, ys, zs, pcolor=True, contour=False, **options):
    """Make a pseudocolor plot.

    xs:
    ys:
    zs:
    pcolor: boolean, whether to make a pseudocolor plot
    contour: boolean, whether to make a contour plot
    options: keyword args passed to pyplot.pcolor and/or pyplot.contour
    """
    _underride(options, linewidth=3, cmap=matplotlib.cm.Blues)

    X, Y = np.meshgrid(xs, ys)
    Z = zs

    x_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
    axes = pyplot.gca()
    axes.xaxis.set_major_formatter(x_formatter)

    if pcolor:
        pyplot.pcolormesh(X, Y, Z, **options)

    if contour:
        cs = pyplot.contour(X, Y, Z, **options)
        pyplot.clabel(cs, inline=1, fontsize=10)


def text(x, y, s, **options):
    """Put text in a figure.

    x: number
    y: number
    s: string
    options: keyword args passed to pyplot.text
    """
    options = _underride(options, verticalalignment='top',
                         horizontalalignment='left')
    pyplot.text(x, y, s, **options)


def config(**options):
    """Configure the plot.

    Pulls options out of the option dictionary and passes them to
    the corresponding pyplot functions.
    """
    names = ['title', 'xlabel', 'ylabel', 'xscale', 'yscale',
             'xticks', 'yticks', 'axis', 'xlim', 'ylim']

    for name in names:
        if name in options:
            getattr(pyplot, name)(options[name])

    # looks like this is not necessary: matplotlib understands text loc specs
    loc_dict = {'upper right': 1,
                'upper left': 2,
                'lower left': 3,
                'lower right': 4,
                'right': 5,
                'center left': 6,
                'center right': 7,
                'lower center': 8,
                'upper center': 9,
                'center': 10,
                }

    loc = options.get('loc', 0)
    # loc = loc_dict.get(loc, loc)

    legend = options.get('legend', True)
    if legend:
        pyplot.legend(loc=loc)


def show(**options):
    """Show the plot.

    For options, see Config.

    options: keyword args used to invoke various pyplot functions
    """
    clf = options.pop('clf', True)
    config(**options)
    pyplot.show()
    if clf:
        clear_figure()


def plotly(**options):
    """Show the plot.

    For options, see Config.

    options: keyword args used to invoke various pyplot functions
    """
    clf = options.pop('clf', True)
    config(**options)
    import plotly.plotly as plotly
    url = plotly.plot_mpl(pyplot.gcf())
    if clf:
        clear_figure()
    return url


def save(root=None, formats=None, **options):
    """Save the plot in the given formats and clears the figure.

    For options, see Config.

    Args:
      root: string filename root
      formats: list of string formats
      options: keyword args used to invoke various pyplot functions
    """
    clf = options.pop('clf', True)
    config(**options)

    if formats is None:
        formats = ['pdf', 'eps']

    try:
        formats.remove('plotly')
        plotly(clf=False)
    except ValueError:
        pass

    if root:
        for fmt in formats:
            save_format(root, fmt)
    if clf:
        clear_figure()


def save_format(root, fmt='eps'):
    """Write the current figure to a file in the given format.

    Args:
      root: string filename root
      fmt: string format
    """
    filename = '%s.%s' % (root, fmt)
    print('Writing', filename)
    pyplot.savefig(filename, format=fmt, dpi=300)


# provide aliases for calling functons with lower-case names
# preplot = PrePlot
# subplot = SubPlot
# clf = Clf
# figure = Figure
# plot = Plot
# scatter = Scatter
# pmf = Pmf
# pmfs = Pmfs
# hist = Hist
# hists = Hists
# diff = Diff
# cdf = Cdf
# cdfs = Cdfs
# contour = Contour
# pcolor = Pcolor
# config = Config
# show = Show
# save = Save


def main():
    """Main."""
    color_iter = _Brewer.ColorGenerator(7)
    for color in color_iter:
        print(color)


if __name__ == '__main__':
    main()
