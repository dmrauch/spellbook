'''
Lower-level functions for creating 2D / bivariate / correlation plots

The high-level functions for creating plots, which make use of the functions
in this module, can be found in :mod:`plot`.

.. todo:: Clean up and complete the documentation of :mod:`spellbook.plot2D`
'''

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from typing import List
from typing import Union

import spellbook as sb



def heatmap_set_annotations_fontsize(
    ax: mpl.axes.Axes,
    fontsize: float) -> None:
    '''
    Set the fontsize for numbers within the heatmap matrix

    Args:
        ax (:class:`matplotlib.axes.Axes`): The axes object
        fontsize: The fontsize to use
    '''
    for child in ax.get_children():
        if isinstance(child, mpl.text.Text):
            child.set_fontsize(fontsize)


def heatmap(data: Union[np.ndarray, pd.DataFrame],
            fig: mpl.figure.Figure,
            grid: mpl.gridspec.GridSpec,
            gridindex: int,
            x: str = '',
            y: str = '',
            normalisation: str = 'count',
            crop = True,
            xlabels: List[str] = None,
            ylabels: List[str] = None,
            ylabels_horizontal: bool = None,
            cmap = 'Blues',
            heatmap_args: dict = {}):
    '''
    Plot the heatmap of a matrix

    The cells of the heatmap are colour-coded according to the values in the
    matrix. The matrix values are also printed in each cell, either in
    absolute numbers or in percentages normalised according to three
    different schemes, set with the **normalisation** parameter.

    Args:
        data (:class:`numpy.ndarray` or :class:`pandas.DataFrame`):
            The :class:`numpy.ndarray` holding the matrix or the
            :class:`pandas.DataFrame` whose correlations should be shown
        fig: The figure on which to draw
        grid: The grid on the figure in which to draw
        gridindex: The index of the grid cell in which to draw
        x: *Optional*. The name of the variable on the x-axis
        y: *Optional*. The name of the variable on the y-axis
        normalisation: *Optional*. How the values shown in the heatmap matrix
            should be normalised

            - ``count``: the matrix values as they are
            - ``norm-all``: the matrix is divided by the sum of all its entries
            - ``norm-row`` or ``norm-true``: each row of the matrix is
              normalised (corresponding to the true labels on the y-axis
              when plotting a confusion matrix)
            - ``norm-col`` or ``norm-pred``: each column of the matrix is
              normalised (corresponding to the predicted labels on the y-axis
              when plotting a confusion matrix)

        crop: *Optional*. When the heatmap is normalised along its rows/columns,
            the summary row/column on the top/right is not used and remains
            empty. When **crop** is set to ``True``, this unused row/column is
            dropped from the heatmap. Otherwise it is included and shows up
            as white space. This can be handy when it is desired to keep
            the cell positions the same for different heatmap plots with
            different normalisations.
        xlabels: *Optional*. The ticklabels for the categories/columns on the
            x-axis
        ylabels: *Optional*. The ticklabels for the categories/rows on the y-axis
        ylabels_horizontal: *Optional*. If set to ``True``, the ticklabels on
            the y-axis are oriented horizontally. Otherwise, their orientation
            is determined automatically by *Matplotlib*
        cmap (*Matplotlib* colourmap name or object, or list of colors):
            *Optional*. The mapping between the values of the heatmap matrix
            and colours that should be used to represent them
        heatmap_args: *Optional*. Dictionary with the arguments and values to
            be passed to :func:`seaborn.heatmap`

    See also:

        This function is usually called through the higher-level functions
        
        - :func:`spellbook.plot.plot_2D`
        - :func:`spellbook.plot.plot_grid_2D`
        - :func:`spellbook.plot.pairplot`
        - :func:`spellbook.plot.plot_confusion_matrix`

        which create the necessary underlying :class:`matplotlib.figure.Figure`
        and :class:`matplotlib.gridspec.GridSpec`.

    .. todo::
        When a SUM column or row is shown, it would be nice to have
        the label of the same axis be centered on the classes only,
        and not on the union of the classes and the SUM column/row
    '''
    
    # will be modifying labels locally, so copy them first
    if not xlabels is None: xlabels = xlabels.copy()
    if not ylabels is None: ylabels = ylabels.copy()

    if isinstance(data, pd.DataFrame):
        assert isinstance(x, str)
        assert isinstance(y, str)

        # crosstab based on https://stackoverflow.com/a/43814116
        crosstab = pd.crosstab(data[y], data[x],
                            margins = False,
                            margins_name = None,
                            normalize = False)
        data = crosstab.values
        xlabels = crosstab.columns.values.tolist()
        ylabels = crosstab.index.values.tolist()
    assert isinstance(data, np.ndarray)

    n = np.sum(data)   # number of datapoints
    nx = data.shape[1] # number of classes
    ny = data.shape[0] # number of classes

    assert len(xlabels) == nx
    assert len(ylabels) == ny
    if normalisation not in ['count', 'norm-all',
                             'norm-row', 'norm-col', 'norm-true', 'norm-pred']:
        raise ValueError("Parameter 'normalisation' has to be 'count', "
            "'norm-all', 'norm-row', 'norm-col', 'norm-true' or 'norm-pred'")

    # determine the dimension of the matrices
    xdim = nx if crop and normalisation in ['norm-pred', 'norm-col'] else nx+1
    ydim = ny if crop and normalisation in ['norm-true', 'norm-row'] else ny+1
    yoffset = 0 if crop and normalisation in ['norm-true', 'norm-row'] else 1

    # add a row and a column for the total counts
    values = np.full(shape=(ydim,xdim), fill_value=-1.0, dtype=float)
    # diag = np.full(shape=(ydim+1,xdim+1), fill_value=-1.0, dtype=float)
    # offdiag = np.full(shape=(ydim+1,xdim+1), fill_value=-1.0, dtype=float)
    totals = np.full(shape=(ydim,xdim), fill_value=-1.0, dtype=float)
    mask = np.full(shape=(ydim,xdim), fill_value=False, dtype=bool)
    for iy in range(ny):
        for ix in range(nx):
            norm = 1.0
            if normalisation == 'norm-all': # normalised across entire matrix
                norm = np.sum(data) / 100.0
            elif normalisation in ['norm-row', 'norm-true']:
                norm = np.sum(data, axis=1)[iy] / 100.0
            elif normalisation in ['norm-col', 'norm-pred']:
                norm = np.sum(data, axis=0)[ix] / 100.0
            mask[iy+yoffset][ix] = True
            if norm > 0:
                values[iy+yoffset][ix] = float(data[iy][ix]) / float(norm)
            else:
                values[iy+yoffset][ix] = 0.0
            # if iy == ix:
            #   diag[iy+yoffset][ix] = values[iy+yoffset][ix]
            # else:
            #   offdiag[iy+yoffset][ix] = values[iy+yoffset][ix]
        if normalisation in ['count', 'norm-all', 'norm-row', 'norm-true']:
            # sum cols along a row
            totals[iy+yoffset][nx] = np.sum(values[iy+yoffset][:nx])
    if normalisation in ['count', 'norm-all', 'norm-col', 'norm-pred']:
        # sum rows along a column
        totals[0][:nx] = np.sum(values, axis=0, where=mask)[:nx]
    if normalisation in ['count', 'norm-all']:
        # total sum
        totals[0][nx] = np.sum(totals[0][:nx])

    # prepare x- and y-labels
    if normalisation in ['count', 'norm-all', 'norm-row', 'norm-true']:
        xlabels.append('SUM')
    if normalisation in ['count', 'norm-all', 'norm-col', 'norm-pred']:
        ylabels.insert(0, 'SUM')
    else:
        if not crop: ylabels.insert(0, '')

    strings = values.tolist()
    totals_strings = totals.tolist()
    for iy in range(ydim):
        for ix in range(xdim):
            if normalisation == 'count':
                strings[iy][ix] = '{:.0f}'.format(strings[iy][ix])
                totals_strings[iy][ix] = '{:.0f}'.format(totals_strings[iy][ix])
            else:
                strings[iy][ix] = '{:.1f}%'.format(strings[iy][ix])
                totals_strings[iy][ix] = '{:.1f}%'.format(totals_strings[iy][ix])

    ax_values = plt.Subplot(fig, grid[gridindex])
    fig.add_subplot(ax_values)
    max_row_totals = -1
    max_col_totals = -1
    if not normalisation in ['norm-pred', 'norm-col']:
        max_row_totals = np.amax(totals[1:][:,nx])
    if not normalisation in ['norm-true', 'norm-row']:
        max_col_totals = np.amax(totals[0][:nx])
    vmax = max(max_col_totals, max_row_totals)
    sns.heatmap(values,
                ax = ax_values,
                mask = (values < 0),
                vmin = 0,
                # vmax = n if normalisation=='count' else 100.0,
                vmax = vmax,
                # square = True, # each cell is square
                linewidths = 1, # separation between cells
                annot = strings,
                fmt = 's',
                cbar = False,
                cmap = cmap,
                **heatmap_args)
    sns.heatmap(totals,
                ax = ax_values,
                mask = (totals < 0),
                annot = totals_strings,
                fmt = 's',
                xticklabels = xlabels,
                yticklabels = ylabels,
                cbar = False,
                cmap = sb.plotutils.cmap_white)
    # [label.set_fontsize("large") for label in ax_values.get_xticklabels()]
    # [label.set_fontsize("large") for label in ax_values.get_yticklabels()]
    if normalisation in ['norm-row', 'norm-true']:
        ax_values.yaxis.get_major_ticks()[0].tick1line.set_visible(False)
    ax_values.set_xlabel(x) # fontsize='x-large'
    ax_values.set_ylabel(y) # fontsize='x-large'
    if ylabels_horizontal: # prevent ticklabels from rotating
        for label in  ax_values.get_yticklabels(): label.set_rotation(0)

    fig.tight_layout()
    return



def categorical_histogram(data, x, y, fig, grid, gridindex,
                          histogram_args: Union[dict, List[dict]] = {},
                          histplot_args: dict = {}
                          ):
    '''
    Multiple horizontal histograms above each other, showing the distribution
    of the continuous x-variable for each value of the categorical y-variable

    .. image:: ../images/categorical-histogram.png
      :width: 500px
      :align: center
    '''

    cats = data[y].unique()
    ncats = len(cats)
    xmin = data[x].max() # on purpose, initialisation
    xmax = data[x].min() # on purpose, initialisation

    # Write the y-variable name on a single common y-axis on the left side.
    # Only the category names are written inside the grids/pads/patches for
    # each histogram.
    ax = plt.Subplot(fig, grid[gridindex])
    fig.add_subplot(ax)
    ax.set_ylabel(y)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # based on https://stackoverflow.com/a/34934631
    finegrid = mpl.gridspec.GridSpecFromSubplotSpec(nrows=ncats, ncols=1,
        subplot_spec=grid[gridindex], wspace=0.0, hspace=0.0)

    ax = []
    if isinstance(histogram_args, dict):
        histogram_args = [histogram_args] * ncats
    for icats in range(ncats):
        if not 'show_stats' in histogram_args[icats]:
            histogram_args[icats]['show_stats'] = False
        ax.append(
            sb.plot1D.histogram(data=data[data[y] == cats[icats]], x=x,
                               fig=fig, grid=finegrid, gridindex=ncats-1-icats,
                               histplot_args = histplot_args,
                               **histogram_args[icats]))
        ax[icats].get_xaxis().set_visible(icats == 0)
        # hide the y-axes for each of the histograms because the y-variable name
        # is only written once on a common y-axis
        ax[icats].get_yaxis().set_visible(False)
        sb.plotutils.categorybox(ax=ax[icats], text='{}'.format(cats[icats]),
            text_args = dict(fontweight='bold'))
        xmin = min(xmin, ax[icats].get_xlim()[0])
        xmax = max(xmax, ax[icats].get_xlim()[1])

    # same x-ranges for all categories
    for icats in range(ncats):
        ax[icats].set_xlim(xmin, xmax)



def violinplot(data, x, y, fig, grid, gridindex):
    '''
    Multiple vertical violins beside each other, showing the distribution
    of the continuous y-variable for each value of the categorical x-variable

    .. image:: ../images/violinplot.png
      :width: 500px
      :align: center
    '''

    ax = plt.Subplot(fig, grid[gridindex])
    fig.add_subplot(ax)
    sns.violinplot(data=data, x=x, y=y, ax=ax, color=sb.plotutils.colours['Blue'])


def scatterplot(data, x, y, ax, bins=25,
                show_scatterplot=True, show_lineplot=True,
                show_axis=True, show_stats=True,
                statsbox_args={}):

  # scatterplot
  if show_scatterplot:
    alpha = 0.4 if show_lineplot else 0.6
    sns.scatterplot(data=data, x=x, y=y, ax=ax, alpha=alpha)

  # lineplot
  if show_lineplot:
    if sb.plotutils.get_data_kind(data[x]) == "cont":
      # bin the x-components
      x_binned = pd.cut(data[x], bins=bins)
      x_centers = [x_binned[i].mid for i in range(len(x_binned))]
      df = pd.DataFrame(data={'xc': x_centers, y: data[y]})
      sns.lineplot(data=df, x='xc', y=y, ax=ax,
                   color='black', ci='sd', err_style='band')
      sns.lineplot(data=df, x='xc', y=y, ax=ax,
                   color='black', ci=95, err_style='bars')
    else:
      # x-components are already 'binned'
      sns.lineplot(data=data, x=x, y=y, ax=ax,
                   color='black', ci='sd', err_style='band')
      sns.lineplot(data=data, x=x, y=y, ax=ax,
                   color='black', ci=95, err_style='bars')

  # stats box
  if show_stats:
    n = len(data[x])
    corrs = data.corr()
    text = sb.plotutils.valuebox_text(
      labels=['count', 'corr'],
      values=[str(n), '{:.3f}'.format(corrs[x][y])]
    )
    sb.plotutils.statsbox(ax, text, **statsbox_args)

  ax.set_xlabel(x)
  if not show_axis: ax.set_axis_off()
