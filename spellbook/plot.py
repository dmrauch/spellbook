'''
High-level functions for creating and saving plots
'''

import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import spellbook as sb



# At some point all the plotting stuff can be organised into proper classes.
# The advantage would be that accessors to elements can be made available in a
# more organised way. Something like
#
#   fig = sb.plot.plot_grid_2D(nrows=2, ncols=5,
#                             data=data, xs=features, ys=target, relative='true')
#   sb.plot2D.heatmap_set_annotations_fontsize(ax=fig.get_axes()[7],
#                                             fontsize='x-small')
#
# would not be necessary anymore and could be replaced with the likes of
#
#   corrs = sb.plot.grid_2D(nrows=2, ncols=5,
#                          data=data, xs=features, ys=target, relative='true')
#   corrs.plots[iRow][iCol].set_annotation_fontsize('x-small')
#
# class PlotBase:
#     # - fig
#     # - ax
#     # - save()
#     pass
#
# class Plot(PlotBase):
#     # - grid(1,1)
#     # - plot: instance of type Barchart, Histogram, Heatmap, ...
#     pass
#
# class PlotGrid(PlotBase):
#     # - grid(n,m)
#     # - axs: [[...], [...]] nested list so that each of them can be accessed
#     # - plots: [[...], [...]] nested list of types Barchart, Histogram, ...
#     pass



def save(fig: mpl.figure.Figure,
         filename: str,
         dpi: int = 200):
    '''
    Save a plot to a file

    Args:
        fig: The figure to plot
        filename: The filename under which to save the plot
        dpi: Optional resolution
    '''
    fig.savefig(filename, dpi=dpi)
    print("Saved plot to file '{}'".format(filename))



def plot_1D(data: pd.DataFrame,
            x: str,
            xlabel: str = None,
            fontsize: float = 12.0,
            figure_args: dict = {},
            barchart_args: dict = {},
            histogram_args: dict = {},
            histplot_args: dict = {},
            statsbox_args: dict = {}
            ) -> mpl.figure.Figure:
    '''
    Create a single univariate plot

    The type of the variable (*categorical* or *continuous*) is determined
    automatically and either :func:`spellbook.plot1D.barchart` or
    :func:`spellbook.plot1D.histogram` is called.

    Args:
        data (:class:`pandas.DataFrame`): The dataset to plot
        x: Name of the variable to plot
        xlabel: *Optional*. Title of the x-axis. If unspecified or set to
            ``None``, the name of the variable, as specified by **x**, will
            be used.
        fontsize: *Optional*. Baseline fontsize for all elements. This is
            probably the fontsize that ``medium`` corresponds to?
        figure_args: *Optional*. Arguments for the creation of the
            :class:`matplotlib.figure.Figure` with
            :func:`matplotlib.pyplot.figure`
        barchart_args: *Optional*. Arguments passed on to
            :func:`spellbook.plot1D.barchart` for categorical data
        histogram_args: *Optional*. Arguments passed on to
            :func:`spellbook.plot1D.histogram` for continuous data
        histplot_args: *Optional*. Arguments for :func:`seaborn.histplot`,
            which is used to draw the plot
        statsbox_args: *Optional*. Arguments passed on by
            `spellbook.plot1D.histogram` to :func:`spellbook.plotutils.statsbox`

    Returns:
        The figure containing the plot
    '''

    if fontsize:
        tmp_fontsize = plt.rcParams['font.size']
        plt.rcParams['font.size'] = fontsize
    fig = plt.figure(tight_layout=True, **figure_args)
    grid = mpl.gridspec.GridSpec(nrows=1, ncols=1, wspace=0.0, hspace=0.0)

    kind = sb.plotutils.get_data_kind(data[x])

    if kind == 'cat':
        sb.plot1D.barchart(data, x=x, fig=fig, grid=grid, gridindex=0,
                          xlabel=xlabel,
                          histplot_args=histplot_args,
                          **barchart_args)

    elif kind == 'ord':
        print('plot1D/ord: TODO / IMPLEMENTATION MISSING')

    elif kind == 'cont':
        sb.plot1D.histogram(data=data, x=x, fig=fig, grid=grid, gridindex=0,
                           xlabel=xlabel,
                           histplot_args=histplot_args,
                           statsbox_args=statsbox_args,
                           **histogram_args)

    if fontsize: plt.rcParams['font.size'] = tmp_fontsize
    return(fig)



def plot_grid_1D(nrows: int, ncols: int,
                 data: pd.DataFrame,
                 target: str = None,
                 features: List[str] = None,
                 xlabels: Union[str, List[str]] = None,
                 fontsize: float = 12.0,
                 figure_args: dict = {},
                 stats: Union[bool, List[bool]] = True,
                 stats_align: Union[str, List[str]] = None,
                 binwidths: Union[float, List[float]] = None,
                 histogram_args: dict = {}
    ) -> mpl.figure.Figure:
    '''
    Create a grid of univariate plots

    The type / visual representation of each variable is determined automatically
    via :func:`spellbook.plotutils.get_data_kind`. Categorical variables are shown as
    barcharts and continuous variables are shown as univariate / 1D histograms.
    Summary statistics boxes can be shown for the histograms.

    .. image:: ../images/plot_grid_1D.png
       :width: 800px
       :align: center

    Args:
        nrows: Number of rows
        ncols: Number of columns
        data (:class:`pandas.DataFrame`): The dataset to plot
        target: *Optional*. The name of the target variable. If specified,
            the target variable will be plotted first and highlighted by
            plotting it in orange. Either **target** or **features** has to
            be specified.
        features: *Optional*. List with the names of the feature
            variables. If specified, the feature variables will be plotted after
            the target variable. Either **target** or **features** has to
            be specified.
        xlabels: *Optional*. The titles of the x-axes. If unspecified or set to
            ``None``, the names of the variables, as specified by **target**
            and **features** will be used.
        fontsize: *Optional*. Baseline fontsize for all elements. This is
            probably the fontsize that ``medium`` corresponds to?
        figure_args: *Optional*. Arguments for the creation of the returned
            :class:`matplotlib.figure.Figure` with
            :func:`matplotlib.pyplot.figure`
        stats: *Optional*. Bool or list of bools that
            indicate if statistics boxes are shown in each plot
        stats_align: *Optional*. List of alignment strings, one for
            each plot
        binwidths: *Optional*. Float or list of floats
            that indicate the binwidth in each plot
        histogram_args: *Optional*. Dictionary of parameters and values
            that are passed to :func:`spellbook.plot1D.histogram`

    Returns:
        Figure containing the grid of plots
    
    Example:

        .. code:: python

            import pandas as pd
            import spellbook as sb
            data = pd.read_csv('dataset.csv')
            plot_vars = sb.plot.plot_grid_1D(2, 4, data,
                target='z', features=['x', 'y'],
                stats=True, stats_align=['tl', 'br', 'tr'])
    '''

    vars = []
    if target: vars = [target]
    if features: vars += features
    assert len(vars) > 0

    if xlabels is None or isinstance(xlabels, str):
        xlabels = [xlabels] * len(vars)
    if isinstance(stats, bool):
        stats = [stats] * len(vars)
    if not type(binwidths) is list:
        binwidths = [binwidths] * len(vars)

    if fontsize:
        tmp_fontsize = plt.rcParams['font.size']
        plt.rcParams['font.size'] = fontsize
    fig = plt.figure(figsize=(3*ncols, 3*nrows), **figure_args)
    grid = mpl.gridspec.GridSpec(nrows=nrows, ncols=ncols)

    for irows in range(nrows):
        for icols in range(ncols):

            i = irows*ncols + icols # vars index
            if i >= len(vars):
                # ax = plt.Subplot(fig, grid[i])
                # ax.axis("off") # blank grid cell, without axes
                continue

            binwidth = binwidths[i] if len(binwidths)>i else None
      
            # statistics box
            stat = stats[i] if len(stats)>i else True
            if stat:
                stats_alignment = stats_align[i] \
                            if (stats_align is not None and len(stats_align)>i) \
                            else "tr"

            histplot_args = {'binwidth': binwidth}
            if target and i==0: histplot_args['color'] = 'C1'

            # categorical variable
            if sb.plotutils.get_data_kind(data[vars[i]]) == 'cat':
                sb.plot1D.barchart(data=data, x=vars[i],
                                  fig=fig, grid=grid, gridindex=i,
                                  xlabel=xlabels[i],
                                  histplot_args=histplot_args)

            # ordinal variable
            elif sb.plotutils.get_data_kind(data[vars[i]]) == 'ord':
                sb.plotutils.not_yet_implemented(fig, grid, i,
                    'plot.plot_grid_1D() / ord')
                # plot = sns.histplot(data=data, x=vars[i], ax=axes[irows][icols], binwidth=binwidth)

                # # statistics box
                # if stat:
                #   sb.plotutils.statsbox(ax=axes[irows][icols],
                #           text=data[vars[i]].describe(percentiles=[]).round(2).to_string(),
                #           alignment=stats_alignment)

            # continuous variable
            else:
                statsbox_args = {}
                if stat: statsbox_args = {'alignment': stats_alignment}
                sb.plot1D.histogram(data=data, x=vars[i],
                                   fig=fig, grid=grid, gridindex=i,
                                   xlabel=xlabels[i],
                                   show_stats=stat,
                                   histplot_args=histplot_args,
                                   statsbox_args=statsbox_args,
                                   **histogram_args)

    fig.tight_layout()
    if fontsize: plt.rcParams['font.size'] = tmp_fontsize
    return(fig)



def plot_2D(data: pd.DataFrame, x: str, y: str,
            relative: bool = False,
            fontsize: float = 12.0,
            figure_args: dict = {},
            heatmap_args: dict = {},
            violinplot_args: dict = {},
            cathist_args: dict = {},
            scatterplot_args: dict = {}
            ) -> mpl.figure.Figure:
    '''
    Create a single bivariate/correlation plot

    The types of the variables (*categorical* or *continuous*) are determined
    automatically and the corresponding 2D plotting function is called:

    - *x* is ``categorical`` and *y* is ``categorical``:
      :func:`spellbook.plot2D.heatmap`
    - *x* is ``categorical`` and *y* is ``continuous``:
      :func:`spellbook.plot2D.violinplot`
    - *x* is ``continuous`` and *y* is ``categorical``:
      :func:`spellbook.plot2D.categorical_histogram`
    - *x* is ``continuous`` and *y* is ``continuous``:
      :func:`spellbook.plot2D.scatterplot`

    Args:
        data (:class:`pandas.DataFrame`): The dataset to plot
        x: Name of the variable to plot on the x-axis
        y: Name of the variable to plot on the y-axis
        relative: Optional, whether or not the heatmaps drawn with
            :func:`spellbook.plot2D.heatmap` should be normalised or not

            - ``True``: heatmap will be column-normalised
              (``normalisation = norm-col``)
            - ``False``: heatmap will be show absolute numbers
              (``normalisation = count``)

        fontsize: *Optional*. Baseline fontsize for all elements.
            This is probably the fontsize that ``medium`` corresponds to?
        figure_args: *Optional*. Arguments for the creation of the
            :class:`matplotlib.figure.Figure` with
            :func:`matplotlib.pyplot.figure`
        heatmap_args: *Optional*. Arguments passed on to
            :func:`spellbook.plot2D.heatmap` for correlations between
            a *categorical* variable on the x-axis and
            a *categorical* variable on the y-axis
        violinplot_args: *Optional*. Arguments passed on to
            :func:`spellbook.plot2D.violinplot` for correlations between
            a *categorical* variable on the x-axis and
            a *continuous* variable on the y-axis
        cathist_args: *Optional*. Arguments passed on to
            :func:`spellbook.plot2D.categorical_histogram` for correlations
            between
            a *continuous* variable on the x-axis and
            a *categorical* variable on the y-axis
        scatterplot_args: *Optional*. Arguments passed on to
            :func:`spellbook.plot2D.scatterplot` for correlations between
            a *continuous* variable on the x-axis and
            a *continuous* variable on the y-axis

    Returns:
        The figure containing the plot

    Examples:

    - simple example

      .. code:: python

         fig = sb.plot.plot_2D(data=data, x='age', y=target, fontsize=14.0)

    - advanced example

      The target variable has two categories and therefore, two histograms
      will be stacked on top of each other. Via the *histogram_args* parameter,
      a list of two dictionaries is passed on to
      :func:`spellbook.plot2D.categorical_histogram` - one dictionary for each
      of the two categories. Each one of the dictionaries is then passed on to
      :func:`spellbook.plot1D.histogram`.

      .. code:: python
  
          fig = sb.plot.plot_2D(
                    data=data, x='age', y=target, fontsize=11.0,
                    cathist_args = {
                        'histogram_args': [
                            dict(
                                show_stats=True,
                                statsbox_args = {'alignment': 'bl'}
                            ),
                            dict(
                                show_stats=True,
                                statsbox_args = {
                                    'y': 0.96,
                                    'text_args': {
                                        # RGBA white with 50% alpha/opacity
                                        'backgroundcolor': (1.0, 1.0, 1.0, 0.5)
                                    }
                                }
                            )
                        ]
                    })
    '''

    if fontsize:
        tmp_fontsize = plt.rcParams['font.size']
        plt.rcParams['font.size'] = fontsize
    fig = plt.figure(tight_layout=True, **figure_args)
    grid = mpl.gridspec.GridSpec(nrows=1, ncols=1, wspace=0.0, hspace=0.0)

    x_kind = sb.plotutils.get_data_kind(data[x])
    y_kind = sb.plotutils.get_data_kind(data[y])

    if x_kind == 'cat' and y_kind == 'cat':
        normalisation = 'norm-col' if relative else 'count'
        sb.plot2D.heatmap(data=data, x=x, y=y,
                         fig=fig, grid=grid, gridindex=0,
                         normalisation=normalisation, **heatmap_args)
    
    elif x_kind == 'cat' and y_kind == 'cont':
        sb.plot2D.violinplot(data=data, x=x, y=y,
                            fig=fig, grid=grid, gridindex=0,
                            **violinplot_args)

    elif x_kind == 'cont' and y_kind == 'cat':
        sb.plot2D.categorical_histogram(data=data, x=x, y=y,
                                       fig=fig, grid=grid, gridindex=0,
                                       **cathist_args)

    else:
        ax = plt.Subplot(fig, grid[0])
        sb.plot2D.scatterplot(data=data, x=x, y=y, ax=ax,
                             **scatterplot_args)
        fig.add_subplot(ax)

    if fontsize: plt.rcParams['font.size'] = tmp_fontsize
    return(fig)



def plot_grid_2D(nrows: int,
                 ncols: int,
                 data: pd.DataFrame,
                 xs: List[str],
                 ys: List[str],
                 relative: bool = False,
                 fontsize: float = 12.0,
                 figure_args: dict = {},
                 heatmap_args: dict = {},
                 violinplot_args: dict = {},
                 cathist_args: dict = {},
                 scatterplot_args: dict = {}
                 ) -> mpl.figure.Figure:
    '''
    Create a grid of bivariate/correlation plots

    Args:
        nrows: Number of rows
        ncols: Number of columns
        data (:class:`pandas.DataFrame`): The dataset to plot
        xs: Names of the variables to plot on the x-axis
        ys: Names of the variables to plot on the y-axis
        relative: *Optional*. Whether or not the heatmaps drawn with
            :func:`spellbook.plot2D.heatmap` should be normalised or not

            - ``True``: heatmap will be column-normalised
              (``normalisation = norm-col``)
            - ``False``: heatmap will be show absolute numbers
              (``normalisation = count``)

        fontsize: *Optional*. Baseline fontsize for all elements.
            This is probably the fontsize that ``medium`` corresponds to?
        figure_args: *Optional*. Arguments for the creation of the
            :class:`matplotlib.figure.Figure` with
            :func:`matplotlib.pyplot.figure`
        heatmap_args: *Optional*. Arguments passed on to
            :func:`spellbook.plot2D.heatmap` for correlations between
            a *categorical* variable on the x-axis and
            a *categorical* variable on the y-axis
        violinplot_args: *Optional*. Arguments passed on to
            :func:`spellbook.plot2D.violinplot` for correlations between
            a *categorical* variable on the x-axis and
            a *continuous* variable on the y-axis
        cathist_args: *Optional*. Arguments passed on to
            :func:`spellbook.plot2D.categorical_histogram` for correlations
            between
            a *continuous* variable on the x-axis and
            a *categorical* variable on the y-axis
        scatterplot_args: *Optional*. Arguments passed on to
            :func:`spellbook.plot2D.scatterplot` for correlations between
            a *continuous* variable on the x-axis and
            a *continuous* variable on the y-axis

    Returns:
        The figure containing the grid of plot
    '''

    if not (type(ys) is list): ys = [ys for i in range(len(xs))]
    assert(len(xs) == len(ys))
    assert(nrows * ncols >= len(xs))

    if fontsize:
        tmp_fontsize = plt.rcParams['font.size']
        plt.rcParams['font.size'] = fontsize
    fig = plt.figure(figsize=(3*ncols, 3*nrows), tight_layout=True,
                     **figure_args)
    grid = mpl.gridspec.GridSpec(nrows=nrows, ncols=ncols)

    for irows in range(nrows):
        for icols in range(ncols):

            i = irows*ncols + icols
            if i >= len(xs):
                ax = plt.Subplot(fig, grid[i])
                ax.axis("off")
                continue

            x_kind = sb.plotutils.get_data_kind(data[xs[i]])
            y_kind = sb.plotutils.get_data_kind(data[ys[i]])

            if x_kind == 'cat' and y_kind == 'cat':
                normalisation = 'norm-col' if relative else 'count'
                sb.plot2D.heatmap(data=data, x=xs[i], y=ys[i],
                                 fig=fig, grid=grid, gridindex=i,
                                 normalisation=normalisation,
                                 **heatmap_args)

            elif x_kind == 'cat' and y_kind == 'cont':
                sb.plot2D.violinplot(data=data, x=xs[i], y=ys[i],
                                    fig=fig, grid=grid, gridindex=0,
                                    **violinplot_args)

            elif x_kind == 'cont' and y_kind == 'cat':
                sb.plot2D.categorical_histogram(data=data, x=xs[i], y=ys[i],
                                               fig=fig, grid=grid, gridindex=i,
                                               **cathist_args)

            else:
                ax = plt.Subplot(fig, grid[0])
                sb.plot2D.scatterplot(data=data, x=x, y=y, ax=ax,
                                      **scatterplot_args)

    if fontsize: plt.rcParams['font.size'] = tmp_fontsize
    return(fig)


def pairplot(data: pd.DataFrame,
             xs: List[str],
             ys: List[str] = None,
             fontsize: float = 12.0,
             histplot_args: dict = {},
             ) -> mpl.figure.Figure:
    '''
    Create a pairplot

    .. image:: ../images/pairplot-3x5.png
       :width: 700px
       :align: center

    The plot does not need to contain the same variables or number of variables
    in x and y. It can be rectangular with any number of rows and any number of
    columns. The subplots with the same variable in x and y are detected
    automatically, no matter where they are located in the pairplot, and instead
    of a 2D/bivariate/correlation plot, the appropriate 1D/univariate
    distribution is shown. This behaviour allows to split a full and possibly
    large pairplot for all variables into arbitrarily-sized separate smaller
    pieces.

    The visual representation of the distributions and correlations is chosen
    automatically depending on the type of random variables (categorical,
    ordinal, continuous).

    Args:
        data (:class:`pandas.DataFrame`): The dataset to plot
        xs: Names of the variables to plot on the x-axis
        ys: *Optional*. Names of the variables to plot on the y-axes.
            If not specified, the same variables will be shown on the x-axes
            and the y-axes.
        fontsize: *Optional*. Baseline fontsize for all elements.
            This is probably the fontsize that ``medium`` corresponds to?
        histplot_args: *Optional*. Arguments for :func:`seaborn.histplot`,
            which is used to draw the histograms

    Returns:
        The figure containing the grid of plots
    '''

    if ys is None: ys = xs
    assert isinstance(data, pd.DataFrame)
    assert isinstance(xs, list)
    assert isinstance(ys, list)
    assert all(isinstance(x, str) for x in xs)
    assert all(isinstance(y, str) for y in ys)

    ncols = len(xs)
    nrows = len(ys)

    if fontsize:
        tmp_fontsize = plt.rcParams['font.size']
        plt.rcParams['font.size'] = fontsize
    fig = plt.figure(figsize=(3*ncols, 3*nrows))
    grid = mpl.gridspec.GridSpec(nrows=nrows, ncols=ncols)

    for iy in range(nrows):
        for ix in range(ncols):

            i = iy*ncols + ix
            x_kind = sb.plotutils.get_data_kind(data[xs[ix]])
            y_kind = sb.plotutils.get_data_kind(data[ys[iy]])
            histplot_args = {}

            if xs[ix] == ys[iy]: # on-diagonal: univariate plots

                histplot_args['color'] = 'C1' # 'gray' # pink

                if x_kind == 'cat':
                    sb.plot1D.barchart(data, x=xs[ix],
                                      fig=fig, grid=grid, gridindex=i,
                                      histplot_args=histplot_args)

                elif x_kind == 'ord':
                    sb.plotutils.not_yet_implemented(fig, grid, i,
                        'plot.pairplot()\non-diagonal / ord')

                elif x_kind == 'cont':
                    sb.plot1D.histogram(data=data, x=xs[ix],
                                       fig=fig, grid=grid, gridindex=i,
                                       show_stats=False,
                                       histplot_args=histplot_args)

                else:
                    assert False # ax.axis("off")

            else: # off-diagonal: bivariate/correlation plots

                if x_kind == 'cat' and y_kind == 'cat':
                    sb.plot2D.heatmap(data=data, x=xs[ix], y=ys[iy],
                                     fig=fig, grid=grid, gridindex=i,
                                     normalisation='norm-col')

                elif x_kind == 'cat' and y_kind == 'cont':
                    sb.plot2D.violinplot(data=data, x=xs[ix], y=ys[iy],
                                        fig=fig, grid=grid, gridindex=i)

                elif x_kind == 'cont' and y_kind == 'cat':
                    sb.plot2D.categorical_histogram(data=data, x=xs[ix], y=ys[iy],
                                                fig=fig, grid=grid, gridindex=i,
                                                histplot_args=histplot_args)

                elif x_kind == 'cont' and y_kind == 'cont':
                    ax = plt.Subplot(fig, grid[i])
                    sb.plot2D.scatterplot(data=data, x=xs[ix], y=ys[iy], ax=ax,
                                         show_lineplot=False,
                                         show_stats=False)
                    fig.add_subplot(ax)

    fig.tight_layout()
    if fontsize: plt.rcParams['font.size'] = tmp_fontsize
    return(fig)


def plot_confusion_matrix(confusion_matrix: tf.Tensor,
                          class_names: List[str],
                          class_ids: List[int] = None,
                          normalisation: str = 'count',
                          figsize: Tuple[float, float] = (5.8, 5.3),
                          crop: bool = True
                          ) -> mpl.figure.Figure:
    '''
    Create a confusion matrix heatmap plot

    .. image:: ../images/confusion-matrix-absolute.png
       :width: 600px
       :align: center

    Both the absolute frequencies as well as the relative frequencies, either
    normalised by the true labels, the predictedlabels or their combinations,
    can be shown. The desired behaviour is specified with the parameter
    ``normalisation``.

    Args:
        confusion_matrix (:class:`tf.Tensor`): The confusion
            matrix
        class_names: List of the class names
        class_ids: Optional, list of IDs for each target class.
            These IDs are shown on the x-axis and, together with the class
            names, on the y-axis.
        normalisation: Optional, indicates if the absolute or relative
            frequencies should be plotted
            
            - ``count``: Numbers of datapoints
            - ``norm-all``: Percentages normalised across all combinations of
              the true and the predicted classes/labels
            - ``norm-true``: Percentages normalised across the true labels
            - ``norm-pred``: Percentages normalised across the predicted classes

        figsize: Optional, size (width, height) of the figure in inches
        crop: Plots with *normalisation* set to ``norm-true``/``norm-pred``
            do not include the *SUM* row/column, respectively. When *crop* is
            set to
            
            - ``True``, the excluded *SUM* row/column is removed from the
              heatmap matrix, thus making it occupy a larger portion of the plot
            - ``False``, the excluded *SUM* row/column is kept empty but still
              included in the heatmap matrix, so as to make each cell appear
              in the same position as with normalisation set to ``count`` or
              ``norm-all``

    Returns:
        The figure containing the plot
    
    See also:
        :func:`tf.math.confusion_matrix`
    '''

    fig = plt.figure(figsize=figsize)
    grid = mpl.gridspec.GridSpec(nrows=1, ncols=1)
    sb.plot2D.heatmap(data = confusion_matrix.numpy(),
                     x = 'predicted labels',
                     y = 'true labels',
                     fig = fig,
                     grid = grid,
                     gridindex = 0,
                     normalisation = normalisation,
                     crop = crop,
                     xlabels = class_ids,
                     ylabels = ['{} - {}'.format(id, name)
                         for id, name in zip(class_ids, class_names)],
                     ylabels_horizontal = True,
                     heatmap_args = dict(square=True))

    fig.tight_layout()
    return(fig)



def parallel_coordinates(
    data: pd.DataFrame,
    features: List[str],
    target: str,
    categories: Dict[str, Dict[int, str]],
    fontsize: float = None,
    shift: float = 0.3
    ) -> mpl.figure.Figure:
    '''
    Parallel coordinates plot

    .. image:: /images/parallel-coordinates.png
       :align: center
       :height: 250px

    Based on `Parallel Coordinates in Matplotlib
    <https://benalexkeen.com/parallel-coordinates-in-matplotlib/>`_,
    but extended to also support categorical variables.

    For categorical variables, a random uniform shift is applied to spread
    the lines in the vicinity of the respective classes. This way, there
    is an indication for the composition of the datapoints in a particular
    class/category in terms of the target labels/classes. Furthermore, the
    shift interval is sized according to the number of datapoints in the
    respective class/category in order to give an impression for how many
    datapoints there are in that class.

    Args:
        data (:class:`pandas.DataFrame`): The dataset to plot
        features: The names of the feature variables
        target: The name of the target variable
        categories: Dictionary holding the category codes/indices and names
            as returned by :func:`spellbook.input.encode_categories`
        fontsize: *Optional*. Baseline fontsize for all elements.
            This is probably the fontsize that ``medium`` corresponds to?
        shift: *Optional*. The half-size of the interval for uniformely
            shifting categorical variables

    .. todo:: Support more than the 10 colours included in *Matplotlib*'s
              tableau colours
    '''

    df = data.copy() # get local copy to protect original data

    target_name = target.replace('_codes', '').replace('_norm', '')
    features.append(target) # 
    feature_names = [f.replace('_codes', '').replace('_norm', '')
        for f in features]
    x = [i for i, _ in enumerate(features)]

    # https://matplotlib.org/stable/gallery/color/named_colors.html
    colours = list(mpl.colors.TABLEAU_COLORS)

    if fontsize:
        tmp_fontsize = plt.rcParams['font.size']
        plt.rcParams['font.size'] = fontsize

    fig, axs = plt.subplots(1, len(features)-1, sharey=False,
        figsize = (1.5*len(features), 5))

    cat_mins = {}
    cat_maxs = {}
    for i, feature in enumerate(features):
        features[i] = feature + '_y'
        if feature_names[i] in categories:
            ncats = len(categories[feature_names[i]])

            # counts / absolute frequencies of each class
            counts = df[feature].value_counts()

            # empty shifting vector, one entry per row/datapoint
            shifts = np.zeros(shape=len(df[feature]))

            # calculate shift for each class
            for index, value in enumerate(df[feature].values):

                # larger shift if class is more frequent
                # -> thickness of line bundle indicates how many datapoints
                #    fall in that particular class
                shifts[index] = shift * (counts[value]-1)/max(5,len(df)-1)

            # apply uniform shifts
            df[feature+'_y'] = 1.0 / (ncats-1) \
                * np.random.uniform(low = df[feature]-shifts,
                                    high = df[feature]+shifts)
            cat_mins[feature+'_y'] = -shift/(ncats-1)
            cat_maxs[feature+'_y'] = (ncats-1+shift)/(ncats-1)
        else:
            df[feature+'_y'] = df[feature]
    cat_min = np.amin(list(cat_mins.values()))
    cat_max = np.amax(list(cat_maxs.values()))

    # Get min, max and range for each column
    # Normalize the data for each column
    min_max_range = {}
    for i, feature in enumerate(features):
        min_val = df[feature].min()
        max_val = df[feature].max()
        val_range = np.ptp(df[feature])
        min_max_range[feature] = [min_val, max_val, val_range]
        if feature_names[i] in categories:
            y = (df[feature]-cat_mins[feature]) / (cat_maxs[feature]-cat_mins[feature])
        else:
            if max_val > 0.0:
                y = df[feature] / max_val
            else:
                y = df[feature]
        df[feature] = cat_min + (cat_max-cat_min) * y

    # Plot each row
    for i, ax in enumerate(axs):
        for idx, row in df.iterrows():
            category = int(row[target])
            ax.plot(x, row[features], colours[category])
        ax.set_xlim([i, i+1])

    # Set the tick positions and labels on y axis for each plot
    # Tick positions based on normalised data
    # Tick labels are based on original data
    def set_ticks_for_axis(dim, ax, ticks):
        min_val, max_val, val_range = min_max_range[features[dim]]

        feature = feature_names[dim]
        if feature in categories:
            tick_labels = categories[feature].values()
            ncats = len(categories[feature])
            ticks = [(cat_index+shift) / (ncats-1+2*shift) 
                for cat_index in categories[feature].keys()]
        else:
            if max_val > 0.0:
                step = max_val / float(ticks-1)
                norm_step = 1.0 / float(ticks-1)
                tick_labels = [round(step * i, 2) for i in range(ticks)]
                ticks = [round(norm_step * i, 2) for i in range(ticks)]
            else:
                step = 1.0 / float(ticks-1)
                norm_step = 1.0 / float(ticks-1)
                tick_labels = [round(step * i, 2) for i in range(ticks)]
                ticks = [round(norm_step * i, 2) for i in range(ticks)]
        ticks = [cat_min + (cat_max-cat_min)*tick for tick in ticks]
        ax.set_yticks(ticks)
        ax.set_yticklabels(tick_labels, backgroundcolor=(1,1,1, 0.7)) # fontweight='bold')
        ax.set_ylim(bottom=-shift-0.08, top=1+shift+0.08)

    for dim, ax in enumerate(axs):
        ax.xaxis.set_major_locator(mpl.ticker.FixedLocator([dim]))
        set_ticks_for_axis(dim, ax, ticks=6)
        ax.set_xticklabels([feature_names[dim]])

    # Move the final axis' ticks to the right-hand side
    ax = axs[-1].twinx()
    dim = len(axs)
    ax.xaxis.set_major_locator(mpl.ticker.FixedLocator([x[-2], x[-1]]))
    set_ticks_for_axis(dim, ax, ticks=6)
    ax.set_xticklabels([feature_names[-2], feature_names[-1]])
    ncats = len(categories[target_name])
    ax.set_ylim(bottom=-shift-0.05, top=1+shift+0.05)

    fig.tight_layout()
    fig.subplots_adjust(wspace=0) # remove space between subplots

    if fontsize: plt.rcParams['font.size'] = tmp_fontsize
    return fig
