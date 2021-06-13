'''
Helper functions used by the other plotting modules
'''

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from typing import List


colours = {
    # spellbook plots
    'Blue': "#5799c6",
    "DarkBlue": "#325973",      # 'value' = 45
    "DarkDarkBlue": "#2c4f66",  # 'value' = 40

    # spellbook logos and icons
    'SpellbookLogoDarkBlue': '#386380', # RGB: (56,99,128), CMYK: (0.56,0.23,0.0,0.5)

    # spellbook documentation / sphinx-book-theme
    'doc-blue': '#0071bc',         # RGB: (  0,113,188), CMYK: (1.0,0.40,0.00, 0.26)
    'doc-orange': '#ff7f0e',       # RGB: (255,127, 14), CMYK: (0.0,0.50,0.95, 0.00)
    'doc-yellow': '#ffc107',       # RGB: (255,193,  7), CMYK: (0.0,0.24,0.97, 0.00)
    'doc-yellow-light': '#fff6dd', # RGB: (255,246,221), CMYK: (0.0,0.04,0.13, 0.00)

    # seaborn colours
    'seaborn-blue': '#6596b8', # violinplot with color=colours['Blue']

    # matplotlib colours
    'C0-Blue':   '#1f77b4',
    'C1-Orange': '#ff7f0e', # RGB: 255,127,14
    'C2-Green':  '#2ca02c',
    'C3-Red':    '#d62728',
}
cmap_white = mpl.colors.ListedColormap(["white"])

legend_valign_dict = {'t': 'upper', 'c': 'center', 'b': 'lower'}
legend_halign_dict = {'l': 'left', 'c': 'center', 'r': 'right'}
legend_vanchor_dict = {'t': 1.0, 'c': 0.5, 'b': 0.0}
legend_hanchor_dict = {'l': 0.0, 'c': 0.5, 'r': 1.0}
def legend_loc(align: str = 'bl'):
    assert len(align) == 2
    return('{} {}'.format(legend_valign_dict[align[0]],
                          legend_halign_dict[align[1]]))
def legend_bbox_to_anchor(align: str = 'bl'):
    assert len(align) == 2
    return((legend_hanchor_dict[align[1]], legend_vanchor_dict[align[0]]))

stats_x_left   = 0.02
stats_x_right  = 0.98
stats_y_top    = 0.98
stats_y_bottom = 0.02
stats_style_dict = {
    "fontfamily": "monospace",
    "alpha": 1.0, #0.8,
    # "backgroundcolor": "white"
}
stats_align_dict = {
    "tl": {"horizontalalignment": "left", "verticalalignment": "top"},
    "tr": {"horizontalalignment": "right", "verticalalignment": "top"},
    "bl": {"horizontalalignment": "left", "verticalalignment": "bottom"},
    "br": {"horizontalalignment": "right", "verticalalignment": "bottom"}
}


def not_yet_implemented(fig, grid, gridindex, message):

    text = message + '\nnot yet implemented'
    print(text.replace('\n', ' - '))
    ax = plt.Subplot(fig, grid[gridindex])
    fig.add_subplot(ax)
    ax.text(0.5, 0.5, text,
            transform=ax.transAxes,
            horizontalalignment='center',
            verticalalignment='center')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


def get_data_kind(var: pd.Series) -> str:
    '''
    Determine the datakind of a :class:`pandas.Series`

    The datakind can be

    - ``cat``: categorical
    - ``ord``: ordinal
    - ``cont``: continuous

    Args:
        var (:class:`pandas.Series`): The column of a :class:`pandas.DataFrame`
    '''

    if isinstance(var, pd.Series):
        if var.dtypes == float:
            return("cont")
        elif var.dtypes == int:
            return("ord")
        elif var.dtypes == object:
            return("cat")

    raise ValueError("Somehow I am missing a type...")


def print_data_kinds(data: pd.DataFrame) -> None:
    '''
    Print datakinds for all columns in the dataframe
    
    The datakinds are determined in :func:`get_data_kind`.

    Args:
        data(:class:`pandas.DataFrame`): *pandas* DataFrame
    '''

    vars = list(data)
    kinds = [get_data_kind(data[var]) for var in vars]
    text = valuebox_text(vars, kinds,
                         label_header='variable',
                         value_header='kind')
    print(text)


def valuebox_text(labels: List[str], values: list,
                  label_header: str = None,
                  value_header: str = None) -> str:
    '''
    Create formatted text with two columns (labels, values)

    Args:
        labels(list[str]): List of the variable names
        values(list): list of the values
        label_header(str): Header of the label column
        value_header(str): Header of the value column
    
    Returns:
        str: The formatted text
    '''

    max_labels = 0
    max_values = 0
    for i in range(len(labels)):
        if type(values[i]) is not str:
            raise ValueError("'values' must contain strings")
        max_labels = max(max_labels, len(labels[i]))
        max_values = max(max_values, len(values[i]))
    if label_header: max_labels = max(max_labels, len(label_header))
    if value_header: max_values = max(max_values, len(value_header))
  
    text = ""
    if label_header or value_header:
        n_labels = max_labels - len(label_header)
        n_values = max_values - len(value_header)
        text += '{}{}   {}{}\n'.format(label_header, ' '*n_labels,
                                       ' '*n_values, value_header)
        text += '{}\n'.format('-'*(max_labels + 3 + max_values))
    for i in range(len(labels)):
        n_labels = max_labels - len(labels[i])
        n_values = max_values - len(values[i])
        text += "{}{}   {}{}{}".format(labels[i], ' '*n_labels,
                                       ' '*n_values, values[i],
                                       '\n' if i < len(labels)-1 else '')
  
    return(text)


def statsbox(ax, text, x=None, y=None, alignment="tr", fontsize='x-small',
             text_args = {}):
  if x is None:
    x = stats_x_left if ("l" in alignment) else stats_x_right # default: right
  if y is None:
    y = stats_y_bottom if ("b" in alignment) else stats_y_top # default: top

  # take stats_style_dict as the default and apply text_args on top
  style_args = stats_style_dict.copy()
  for key, value in text_args.items():
      style_args[key] = value

  statbox = ax.text(x, y, text, transform=ax.transAxes, fontsize=fontsize,
                    **style_args, **stats_align_dict[alignment])
  return(statbox)


def categorybox(ax, text, x=None, y=None, alignment="tl", text_args={}):
    '''
    Args:
        ax
        text
        x
        y
        alignment
        text_args(dict): Dictionary of keyword arguments that are passed on to
            :meth:`matplotlib.axes.Axes.text`.

            For example:

            - ``fontsize`` can be ``xx-small``, ``x-small``, ``small``,
              ``medium``, ``large``, ``x-large`` or ``xx-large``
    '''

    if x is None:
        x = stats_x_left if ("l" in alignment) else stats_x_right
    if y is None:
        y = stats_y_bottom if ("b" in alignment) else stats_y_top

    categorybox = ax.text(x, y, text, transform=ax.transAxes, # fontsize=fontsize,
                          **stats_align_dict[alignment],
                          **text_args,
                          bbox = dict(boxstyle='square,pad=0.0',
                                      color='white')) # alpha=0.5
                          # based on https://stackoverflow.com/a/29127933
    return(categorybox)
