'''
Lower-level functions for creating 1D / univariate plots

The high-level functions for creating plots, which make use of the functions
in this module, can be found in :mod:`plot`.
'''

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import tensorflow as tf

from typing import Union

import spellbook as sb


def barchart(data: pd.DataFrame,
             x: str,
             fig: mpl.figure.Figure,
             grid: mpl.gridspec.GridSpec,
             gridindex: int,
             xlabel: str = None,
             histplot_args={}) -> mpl.axes.Axes:
    '''
    Draw a vertical barchart

    Args:
        data: The dataset to plot
        x: Name of the variable to plot
        fig: The figure on which to draw
        grid: The grid on the figure in which to draw
        gridindex: The index of the grid cell in which to draw
        histplot_args: *Optional*. Arguments for :func:`seaborn.histplot`

    Returns:
        The axes object drawn. The axes object is already added to the figure,
        but it is returned as well so that it can be caught for easier access
        and manipulation
    '''

    ax = plt.Subplot(fig, grid[gridindex])
    fig.add_subplot(ax)

    plot = sns.histplot(data=data, x=x, ax=ax, **histplot_args)

    # matplotlib draws lazily, need xticklabels before plot is shown
    # based on https://stackoverflow.com/a/41124884
    fig.canvas.draw()

    # annotations in the histogram bars
    # - category labels
    # - counts and frequencies
    xvals = plot.axes.get_xticks()
    xlabels = [xlabel.get_text() for xlabel in plot.axes.get_xticklabels()]
    yvals = data[x].value_counts(normalize=False)
    yfreqs = data[x].value_counts(normalize=True)
    ymax = plot.axes.get_ylim()[1]
    plot.axes.set_xticklabels([]) # remove xticklabels under the axis to avoid overlaps
    for iB in range(len(xvals)):
        plot.annotate("{}".format(xlabels[iB]),
                    xy=(xvals[iB], 0.05*ymax),
                    horizontalalignment="center", rotation="vertical")
        plot.annotate("{} / {:.2f}%".format(yvals[xlabels[iB]], 100.0*yfreqs[xlabels[iB]]),
                    xy=(xvals[iB], 0.90*ymax),
                    horizontalalignment="center",
                    verticalalignment="top",
                    rotation="vertical", **sb.plotutils.stats_style_dict)

    if xlabel is None: xlabel = x
    ax.set_xlabel(xlabel)
    return(ax)



def boxplot(data, ax, x=None, y=None, orient='v',
            lw_mean=6.0, lw_median=4.0,
            show_axis=True, show_stats=True):
    '''
    Draw a boxplot

    .. todo:: Document :func:`spellbook.plot1D.boxplot`
    '''

    # sns.boxplot(data=data, y='data')
    sns.boxplot(data=data, x=x, y=y, ax=ax, orient=orient)
    stats = sb.stat.describe(data)
  
    xmin = -0.4; xmax = 0.4
    ymin = -0.4; ymax = 0.4
  
    if orient == 'v':
        ax.get_xaxis().set_visible(False)
        stats_text = sb.stat.describe_text(data[y])
    
        # standard deviation
        ax.add_patch(mpl.patches.Rectangle(xy=(xmin, stats['mean']-stats['std']),
                    width=xmax-xmin, height=2.0*stats['std'],
                    color='grey', alpha=0.4, zorder=0))
        # median
        ax.hlines(stats['median'], xmin=xmin, xmax=xmax,
                  color=sb.plotutils.colours["DarkDarkBlue"], linewidth=lw_median, zorder=4)
        # mean
        ax.hlines(stats['mean'], xmin=xmin, xmax=xmax,
                  color='black', linewidth=lw_mean, zorder=4)
        # CI(mean)
        ax.vlines(x=0.0,
                  ymin=stats['mean']-stats['CI_mean'],
                  ymax=stats['mean']+stats['CI_mean'],
                  color='black', linewidth=lw_mean, zorder=4)
    
    elif orient == 'h':
        ax.get_yaxis().set_visible(False)
        stats_text = sb.stat.describe_text(data[x])
    
        # standard deviation
        ax.add_patch(mpl.patches.Rectangle(xy=(stats['mean']-stats['std'], ymin),
                    width=2.0*stats['std'], height=ymax-ymin,
                    color='grey', alpha=0.4, zorder=0))
        # median
        ax.vlines(x=stats['median'], ymin=ymin, ymax=ymax,
                  color=sb.plotutils.colours["DarkDarkBlue"], linewidth=lw_median, zorder=4)
        # mean
        ax.vlines(x=stats['mean'], ymin=ymin, ymax=ymax,
                  color='black', linewidth=lw_mean, zorder=4)
        # CI(mean)
        ax.hlines(y=0.0,
                  xmin=stats['mean']-stats['CI_mean'],
                  xmax=stats['mean']+stats['CI_mean'],
                  color='black', linewidth=lw_mean, zorder=4)
  
    if show_stats:
        sb.plotutils.statsbox(ax=ax, text=stats_text)
    if not show_axis: ax.set_axis_off()


def histogram(data: Union[pd.DataFrame, np.ndarray, tf.Tensor],
              fig: mpl.figure.Figure,
              grid: mpl.gridspec.GridSpec,
              gridindex: int,
              x: str = None,
              xlabel: str = None,
              CL_mean: int = 95,
              lw_mean: float = 6.0,
              lw_median: float = 4.0,
              lw_quartiles: float = 2.0,
              show_histogram: bool = True,
              show_decorations: bool = True,
              show_mean: bool = True,
              show_std: bool = True,
              show_sem: bool = True,
              show_median: bool = True,
              show_quartiles: bool = True,
              show_min: bool = True,
              show_max: bool = True,
              show_axis: bool = True,
              show_stats: bool = True,
              histplot_args: dict = {},
              statsbox_args: dict = {},
              ymin: float = None,
              ymax: float = None,
              yscale: str = None
              ) -> mpl.axes.Axes:
    '''
    Draw a histogram

    .. image:: ../images/histogram.png
       :width: 500px
       :align: center

    Besides the pure histogram itself, various lines, bands and bars can be
    drawn that indicate a range of descriptive statistics:

    - a black line for the mean
    - a grey band for the standard deviation
    - a black error bar indicating the confidence interval of the standard error
      of the mean at a desired level
    - a thick dark blue line for the median
    - thin dashed lines for the 25% and 75% quartiles
    - thin solid lines for the minimum and maximum values

    Args:
        data (:class:`pandas.DataFrame`, :class:`numpy.ndarray` or \
        :class:`tf.Tensor`): The data to plot
        fig: The figure on which to draw
        grid: The grid on the figure in which to draw
        gridindex: The index of the grid cell in which to draw
        x: *Optional*. The name of the variable to plot
        xlabel: *Optional*. The x-axis label.
        CL_mean: *Optional*. The confidence level in percent for the uncertainty
            bar around the mean giving the standard error of the mean
        lw_mean: *Optional*. The linewidth of the mean
        lw_median: *Optional*. The linewidth of the median
        lw_quartiles: *Optional*. The linewidth of the quartiles
        show_histogram: *Optional*. If the histogram should be shown
        show_decorations: *Optional*. If set to ``False``, none of the lines
            and errors bands/bars indicating the descriptive statistics will
            be shown
        show_mean: *Optional*. If the mean should be shown
        show_std: *Optional*. If the standard deviation should be shown
        show_sem: *Optional*. If the standard error of the mean should be shown
        show_median: *Optional*. If the median should be shown
        show_quartiles: *Optional*. If the quartiles should be shown
        show_min: *Optional*. If the minimum value / 0% percentile should be
            shown
        show_max: *Optional*. If the minimum value / 1000% percentile should
            be shown
        show_axis: *Optional*. If the axes, including labels, ticks and
            ticklabels should be shown
        show_stats: *Optional*. If a box giving the values of the descriptive
            statistics should be shown
        histplot_args: *Optional*. Arguments for :func:`seaborn.histplot`
        statsbox_args: *Optional*. Arguments for
            :func:`spellbook.plotutils.statsbox`
        ymin: *Optional*. The lower view limit of the y-axis.
        ymax: *Optional*. The upper view limit of the y-axis.
        yscale: *Optional*. The type of scale used for the y-axis. See
            :meth:`matplotlib.axes.Axes.set_yscale`.

    Returns:
        :class:`matplotlib.axes.Axes`: The axes object drawn. The axes object
        is already added to the figure, but it is returned as well so that it
        can be caught for easier access and manipulation.
    
    Examples:

    - plotting a :class:`pandas.DataFrame`

      .. testcode::

         import matplotlib as mpl
         import numpy as np
         import pandas as pd
         import spellbook as sb

         fig = mpl.pyplot.figure(tight_layout=True)
         grid = mpl.gridspec.GridSpec(nrows=1, ncols=1)

         x = np.random.normal(size=1000)
         data = pd.DataFrame({'x': x})
         ax = sb.plot1D.histogram(data, fig, grid, gridindex=0, x='x')

    - plotting a :class:`numpy.ndarray` (:class:`tf.Tensor` analogous)

      .. testcode::

         import matplotlib as mpl
         import numpy as np
         import spellbook as sb

         fig = mpl.pyplot.figure(tight_layout=True)
         grid = mpl.gridspec.GridSpec(nrows=1, ncols=1)

         x = np.random.normal(size=1000)
         ax = sb.plot1D.histogram(x, fig, grid, gridindex=0,
                                 xlabel='x-axis label')

    '''

    if isinstance(data, pd.DataFrame): data = data[x]

    # s = stats.describe(data[x])
    s = stats.describe(data)
    n = s.nobs
    mean = s.mean
    std = np.sqrt(s.variance) # standard deviation
    t = np.fabs(stats.t(df=n-1).ppf((1-CL_mean/100.0)/2.0))
    sem = stats.sem(data) # standard error of the mean
    CI_mean = t * sem
  
    med = np.median(data)
    quantiles = np.quantile(data, [0.0, 0.25, 0.5, 0.75, 1.0])
  
    ax = plt.Subplot(fig, grid[gridindex])
    fig.add_subplot(ax)
  
    if xlabel is None: xlabel = x
    if show_histogram:
        sns.histplot(data=data, ax=ax, **histplot_args)
        y_max = ax.get_ylim()[1]
    else:
        ax.get_yaxis().set_visible(False)
        y_max = 1.0

    # show the lines, bars and bands indicating the descriptive statistics
    if show_decorations:

        # standard deviation
        if show_std: ax.add_patch(mpl.patches.Rectangle(xy=(mean-std, 0),
                                width=2.0*std, height=y_max,
                                color='grey', alpha=0.4, zorder=0))
    
        # median, quartiles, min, max
        if show_median:
            ax.axvline(med,
                       color=sb.plotutils.colours["DarkDarkBlue"],
                       linewidth=lw_median)
        if show_min:
            ax.axvline(quantiles[0],
                       color=sb.plotutils.colours["DarkDarkBlue"],
                       linewidth=lw_quartiles)
        if show_quartiles:
            ax.axvline(quantiles[1],
                       color=sb.plotutils.colours["DarkDarkBlue"],
                       linewidth=lw_quartiles,
                       linestyle='--')
            ax.axvline(quantiles[3],
                       color=sb.plotutils.colours["DarkDarkBlue"],
                       linewidth=lw_quartiles,
                       linestyle='--')
        if show_max:
            ax.axvline(quantiles[4],
                       color=sb.plotutils.colours["DarkDarkBlue"],
                       linewidth=lw_quartiles)

        # mean and standard error of the mean
        if show_mean: ax.axvline(mean, color='black', linewidth=lw_mean)
        if show_sem: ax.hlines(y_max/2.0, xmin=mean-CI_mean, xmax=mean+CI_mean,
                               color='black', linewidth=lw_mean, zorder=4)
    
    # stats box
    if show_stats:
        sb.plotutils.statsbox(ax, text=sb.stat.describe_text(data),
                             **statsbox_args)
  
    ax.set_ymargin(0.0)
    ax.set_xlabel(xlabel)
    if ymax: ax.set_ylim(top=ymax)
    if ymin: ax.set_ylim(bottom=ymin)
    if yscale: ax.set_yscale(yscale)
    if not show_axis: ax.set_axis_off()
  
    return(ax)
