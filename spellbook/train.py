'''
Functions for model training and validation
'''

from __future__ import annotations   # for type hinting the enclosing class
                                     # https://stackoverflow.com/a/33533514
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import sklearn.metrics
import tensorflow as tf

from typing import List
from typing import Union

import spellbook as sb



class ModelSavingCallback(tf.keras.callbacks.Callback):
    '''
    Callback that intermittently saves the full model during training

    The model is saved as a folder intermittently in configurable steps
    during training as well as at the end of the training. The foldername
    is kept unchanged as training goes on, thus always replacing the
    last/old model with the current/new state.
    '''

    def __init__(self,
                 foldername: str = 'model',
                 step: int = 10
                ) -> ModelSavingCallback:
        '''
        Callback that intermittently saves the full model during training

        Args:
            foldername: The model is saved as a folder with this name
            step: The model is saved every *step* number of epochs
        
        Returns:
            An instance of the callback, which is then to be added to the
            *callbacks* parameter of :func:`tensorflow.keras.Model.fit`
        '''
        self.foldername = foldername
        self.step = step

    def on_epoch_end(self, epoch: int, logs: dict) -> None:
        '''
        Called at the end of every epoch: Save the model every *self.step*
        epochs

        Args:
            epoch (int): Index of the epoch
            logs (dict): Metrics for this training epoch as well as the
                validation epoch if validation is performed. Validation result
                keys are prefixed with ``'val_'``.
        '''
        if epoch + 1 == self.params['epochs']: return
        if (epoch + 1) % self.step == 0:
            self.model.save(self.foldername)
            print("Epoch {}/{}: Saved model to folder '{}'".format(
                epoch+1, self.params['epochs'], self.foldername))
    
    def on_train_end(self, logs: dict) -> None:
        '''
        Called at the end of the training: Save the model

        Args:
            logs (dict): Currently the output of the last call to
                :func:`on_epoch_end()` is passed to this argument for this
                method but that may change in the future.
        '''
        self.model.save(self.foldername)
        print("Training finished after {} epochs:".format(
                self.params['epochs']) \
              + " Saved model to folder '{}'".format(self.foldername))



class ROCPlot:
    '''
    Plot containing Receiver Operator Characteristic (ROC) curves and working
    points (WP)

    .. image:: ../images/roc.png
       :width: 550px
       :align: center
    '''

    def __init__(self):
        self.curves = {}
        # default figsize is (6.4, 4.8) [inches]
        self.fig, self.ax = plt.subplots(figsize=(5.8, 5.3), tight_layout=True)
        self.WP_legend_lines = []
        self.WP_legend_labels = []


    def __str__(self):
        text = 'ROCPlot'
        if len(self.curves) == 0:
            text += ' contains no curves'
        else:
            text += ' contains'
            for name in self.curves.keys():
                text += '\n- {}'.format(name)
        return(text)


    def __iadd__(self, other: ROCPlot) -> ROCPlot:
        '''
        The ``+=`` operator
        '''
        for name, curve in other.curves.items():
            if not (name in self.curves.keys()):
                self.add_curve(name, curve['labels'], curve['predictions'],
                               curve['plot_args'])
        return(self)


    def get_legend_lines(self) -> List[mpl.lines.Line2D]:
        '''
        Get the lines representing the ROC curves to be shown in the legend

        Returns:
            [:class:`matplotlib.lines.Line2D`]: Lines representing the ROC
            curves to be shown in the legend
        '''
        lines = []
        for curve in self.curves.values():
            lines.append(curve['line'])
        return(lines)


    def get_legend_labels(self) -> List[str]:
        '''
        Get the labels / texts for the ROC curves to be shown in the legend

        Returns:
            [str]: Labels / texts for the ROC curves to be shown in the legend
        '''
        labels = []
        for name, curve in self.curves.items():
            labels.append('{}: {:.3f}'.format(name, curve['auc']))
        return(labels)


    def add_curve(
        self,
        name: str,
        labels: Union[List[float], np.ndarray, tf.Tensor],
        predictions: Union[List[float], np.ndarray],
        plot_args: dict = {}
        ) -> None:
        '''
        Add a ROC curve

        Internally it uses :func:`sklearn.metrics.roc_auc_score` to calculate
        the ROC curve from the true labels and the predictions / classifier
        outputs.

        Args:
            name (str): The name of the ROC curve
            labels ([float] / numpy.ndarray(ndim=1)): The target labels for
                the datapoints
            predictions ([float] / numpy.ndarray(ndim=1)): The sigmoid-
                activated predictions for the datapoints. Note that not the
                predicted labels but rather the sigmoid-activated predictions
                should be used so that by scanning different thresholds the
                different true positive / false positive rates can be
                determined.
            plot_args (dict): Arguments to be passed to
                :meth:`matplotlib.axes.Axes.plot`
        '''

        if name in self.curves.keys():
            raise KeyError("A curve with the name '{}' exists already".format(name))

        # based on https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#plot_the_roc
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels, predictions)

        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
        auc = sklearn.metrics.roc_auc_score(labels, predictions)

        line = self.ax.plot(100*fpr, 100*tpr,
                            # label = '{}: {:.3f}'.format(name, auc),
                            linewidth = 2,
                            **plot_args)
        assert(len(line) == 1)
        self.curves[name] = dict(labels=labels,
                                 predictions=predictions,
                                 plot_args=plot_args,
                                 fpr = fpr,
                                 tpr = tpr,
                                 thresholds = thresholds,
                                 line=line[0],
                                 auc=auc)


    def rename_curve(self, old_name: str, new_name: str) -> None:
        '''
        Change the name of a ROC curve

        Args:
            old_name(str): The old name
            new_name(str): The new name
        '''
        if old_name in self.curves.keys():
            self.curves[new_name] = self.curves.pop(old_name)


    def remove_curve(self, name: str) -> None:
        '''
        Remove a ROC curve

        Args:
            name(str): The name of the curve to be removed
        '''
        if name in self.curves.keys():
            self.curves.pop(name)


    def get_WP(
        self,
        name:str,
        threshold: Union[float, List[float]] = None,
        FPR: Union[float, List[float]] = None,
        TPR: Union[float, List[float]] = None,
        method: str = 'interpolate'
        ) -> Union[dict, List[dict]]:
        '''
        Find working point(s) at specified classifier threshold(s), TPR(s) or
        FPR(s)

        Either the classifier threshold, the TPR or the FPR parameter can be
        specified to determine one or more matching working points (WP), but not
        more than one of the three at the same time.

        Args:
            name (str): Name of the curve from which to determine the WP(s)
            threshold (float / [float]): Threshold/cut value on the classifier's
                sigmoid-activated output to separate the two classes
            FPR (float / [float]): Target FPR(s) to match
            TPR (float / [float]): Target TPR(s) to match
            method (``interpolate`` / ``nearest``): Method for determining the
                working point(s)

                - ``interpolate``: find the two points that enclose the target
                  value and interpolate linearly between them
                - ``nearest``: find the point that is nearest to the target
                  value

        Returns:
            dict / [dict]: If `threshold`/`FPR`/`TPR` is a scalar, then a dict
            containing the matching WP with its threshold, FPR and TPR
            is returned. If `threshold`/`FPR`/`TPR` is a list, then a list of
            dicts is returned, with each entry in the list corresponding to one
            WP.
        
        .. _wp-example:

        Example:

            - Get a single working point by specifying a classifier threshold
              and draw it

              .. code:: python
  
                 roc = sb.train.ROCPlot()
                 roc.add_curve('1000 epochs (testing)',
                               test_labels, test_predictions.numpy(),
                               plot_args = dict(color='C0', linestyle='-'))
                 WP = roc.get_WP('1000 epochs (testing)', threshold=0.5)
                 roc.draw_WP(WP, linecolor='C0')
                 fig = roc.plot()
                 sb.plot.save(fig, 'roc.png')
            
            - Get multiple working points by specifying FPRs and TPRs and draw
              them

              .. code:: python
                 
                 roc = sb.train.ROCPlot()
                 roc.add_curve('1000 epochs (testing)',
                               test_labels, test_predictions.numpy(),
                               plot_args = dict(color='C0', linestyle='-'))
                 WPs1 = roc.get_WP('1000 epochs (testing)', FPR=[0.1, 0.2])
                 WPs2 = roc.get_WP('1000 epochs (testing)', TPR=[0.8, 0.9])
                 roc.draw_WP(WPs1+WPs2, linecolor=['C0', 'C1', 'C2', 'black'])
                 sb.plot.save(roc.plot(), 'roc.png')
        '''

        find_threshold = not (threshold is None)
        find_FPR = not (FPR is None)
        find_TPR = not (TPR is None)

        # WP(s) defined either by threshold, TPR or FPR, but not more than one
        assert(find_threshold + find_FPR + find_TPR == 1)

        assert method in ['interpolate', 'nearest']

        if not name in self.curves.keys():
            raise KeyError("There is no curve with the name '{}'".format(name))

        def find_closest(A, target):
            '''
            In an sorted array, find the index of the entry closest to a target
            value

            Args:
                A(:class:`numpy.ndarray`): Must be sorted, order does not
                    matter
                target(float): The target value to approximate

            Returns:
                int: The index such that A[index] approximates to target
            '''
            assert(A.ndim == 1)
            assert(A.shape[0] > 1)

            # assume A is ordered, figure out if ascending or descending
            # https://stackoverflow.com/a/47004533
            ascending = (np.all(np.diff(A) >= 0))

            if ascending:
                return(find_closest_ascending(A, target))
            else:
                A_flipped = np.flip(A)
                index_flipped = find_closest_ascending(A_flipped, target)
                index = A.shape[0] - 1 - index_flipped
                return(index)

        def find_closest_ascending(A, target):
            '''
            In an ascending array, find the index of the entry closest to a
            target value

            As given in https://stackoverflow.com/a/8929827

            Args:
                A(:class:`numpy.ndarray`): Must be sorted in ascending order
                target(float): The target value to approximate

            Returns:
                int: The index such that A[index] approximates to target
            '''
            # check ascending order: https://stackoverflow.com/a/47004533
            ascending = (np.all(np.diff(A) >= 0))
            assert(ascending)

            # indentify index/indices of closest entry/entries
            idx = A.searchsorted(target)
            idx = np.clip(idx, 1, len(A)-1)
            left = A[idx-1]
            right = A[idx]
            idx -= target - left < right - target
            return idx
        
        def safe_index(index):
            return((0, 1) if index == 0 else (index-1, index))

        def find_enclosing(A, target):
            assert(A.ndim == 1)
            assert(A.shape[0] > 1)

            # assume A is ordered, figure out if ascending or descending
            # https://stackoverflow.com/a/47004533
            ascending = (np.all(np.diff(A) >= 0))

            if ascending:
                return(find_enclosing_ascending(A, target))
            else:
                A_flipped = np.flip(A)
                index_flipped = find_enclosing_ascending(A_flipped, target)
                index0 = A.shape[0] - 1 - index_flipped[0]
                index1 = A.shape[0] - 1 - index_flipped[1]
                return((index0, index1))

        def find_enclosing_ascending(A, target):
            # check ascending order: https://stackoverflow.com/a/47004533
            ascending = (np.all(np.diff(A) >= 0))
            assert(ascending)

            # indentify index/indices of so that target is in (A[idx-1], A[idx]]
            idx = A.searchsorted(target)
            if idx.ndim == 0:
                return(safe_index(idx))
            # if isinstance(idx, np.ndarray):
            elif idx.ndim == 1:
                i0 = []
                i1 = []
                for i in range(len(idx)):
                    index = safe_index(idx[i])
                    i0.append(index[0])
                    i1.append(index[1])
                return((np.array(i0), np.array(i1)))
            else:
                raise TypeError("'idx' has wrong dimension: {}".format(
                    idx.ndim))

        def interpolation(s, s0, s1, p0, p1):
            alpha = (s1-s)/(s1-s0)
            p = alpha*p0 + (1-alpha)*p1
            return(p)

        thresholds = self.curves[name]['thresholds']
        fpr = self.curves[name]['fpr']
        tpr = self.curves[name]['tpr']
        n = len(thresholds)
        assert(n == len(fpr))
        assert(n == len(tpr))

        if method == 'interpolate':

            if find_threshold:
                i0, i1 = find_enclosing(thresholds, threshold)
                s = threshold; s0 = thresholds[i0]; s1 = thresholds[i1]
            elif find_FPR:
                i0, i1 = find_enclosing(fpr, FPR)
                s = FPR; s0 = fpr[i0]; s1 = fpr[i1]
            elif find_TPR:
                i0, i1 = find_enclosing(tpr, TPR)
                s = TPR; s0 = tpr[i0]; s1 = tpr[i1]
            p0 = np.array([thresholds[i0], fpr[i0], tpr[i0]])
            p1 = np.array([thresholds[i1], fpr[i1], tpr[i1]])
            p = interpolation(s, s0, s1, p0, p1)

            if p.ndim == 1:
                return(dict(threshold=p[0], FPR=p[1], TPR=p[2]))
            elif p.ndim == 2:
                return([dict(threshold=p[0][i], FPR=p[1][i], TPR=p[2][i])
                        for i in range(p.shape[1])])
            else:
                raise TypeError("'p' has wrong dimension: {}".format(p.ndim))

        elif method == 'nearest':

            if find_threshold:
                index = find_closest(thresholds, threshold)
            elif find_TPR:
                index = find_closest(tpr, TPR)
            elif find_FPR:
                index = find_closest(fpr, FPR)

            if index.ndim == 0:
                return(dict(threshold=thresholds[index],
                            FPR=fpr[index], TPR=tpr[index]))
            elif index.ndim == 1:
                return([dict(threshold=thresholds[i], FPR=fpr[i], TPR=tpr[i])
                        for i in index])
            else:
                raise TypeError("'index' is of wrong type or shape")


    def draw_WP(
        self,
        WP: Union[dict, List[dict]],
        line: Union[bool, List[bool]] = True,
        linestyle: Union[str, List[str]] = '-',
        linecolor: Union[str, List[str]] = 'C0', #['C2', 'C0', 'black'],
        info: Union[bool, List[int]] = None,
        highlight: Union[bool, int] = None
        ) -> None:
        '''
        Draw one or more working points and indicate their classifier thresholds
        as well as true positive and false positive rates

        Args:
            WP (dict / [dict]): One or more working points as returned by
                :func:`get_WP`
            linestyle (str / [str], optional): Linestyle(s) for drawing the
                horizontal and vertical lines designating the working point(s)
            info (bool / [int], optional): Whether or not the parameters of the
                WP(s) should be shown on the graph. If ``None`` or ``True``,
                then all WPs are included. If ``False`` or ``[]``, none are
                given. If a list of integers is given, then the WPs with the
                corresponding indices are shown.
            highlight (bool / int, optional): Whether or not and which WP and
                its parameters should be highlighted. If a single WP is given,
                then highlight should be a bool. If multiple WPs are given, it
                should be an integer specifying the index of the WP to
                highlight.
        
        Some examples of how to calculate and draw working points are given
        :ref:`here <wp-example>` in :func:`get_WP`.
        '''

        assert isinstance(WP, dict) or isinstance(WP, list)
        if isinstance(WP, dict):
            assert isinstance(line, bool)
            assert isinstance(linestyle, str)
            assert isinstance(linecolor, str)
            assert info is None or isinstance(info, bool)
        if isinstance(WP, list):
            assert len(WP)>0
            assert isinstance(WP[0], dict)

        # needed here for the transforms to work
        self.fig.canvas.draw()

        if isinstance(WP, dict): WP = [WP]
        if isinstance(line, bool): line = [line]*len(WP)
        if isinstance(linestyle, str): linestyle = [linestyle]*len(WP)
        if isinstance(linecolor, str): linecolor = [linecolor]*len(WP)

        for i in range(len(WP)):
            wp = WP[i]
            self.ax.scatter(x=wp['FPR']*100.0, y=wp['TPR']*100.0, # s=20.0
                            color='red', zorder=3)

            if highlight or (type(highlight)==int and highlight==i):
                color = 'red'
            else:
                color = linecolor[i] # 'grey'

            # draw line if so desired
            if line[i]:
                hline = self.ax.axhline(y=wp['TPR']*100.0, zorder=-1,
                                        color=color, linestyle=linestyle[i],
                                        linewidth=1.0)
                self.ax.plot(
                    [wp['FPR']*100.0, wp['FPR']*100.0], [0.0, wp['TPR']*100.0],
                    zorder=-1, color=color,
                    linestyle=linestyle[i], linewidth=1.0)
                if info is None or info == True \
                                or (type(info)==list and len(info)>i and info[i]):
                    self.WP_legend_lines.append(hline)
                    self.WP_legend_labels.append(
                        '{:.4f} → FPR: {:.1f}%, TPR: {:.1f}%'.format(
                            wp['threshold'], wp['FPR']*100.0, wp['TPR']*100.0))

        if len(self.WP_legend_labels):
            # Reverse the lines and labels array so that the most recently
            # added working point is displayed above the others
            self.legend_WP = self.ax.legend(handles=self.WP_legend_lines[::-1],
                                            labels=self.WP_legend_labels[::-1],
                                            fontsize='x-small',
                                            labelcolor='linecolor',
                                            loc='lower right')


    def plot(self,
        xmin: float = 0.0,
        xmax: float = 100.0,
        ymin: float = 0.0,
        ymax: float = 102.0
        ) -> mpl.figure.Figure:
        '''
        Plot one or more ROC curves and working points to a figure object

        Args:
            xmin: Lower end of the x-axis
            xmax: Upper end of the x-axis
            ymin: Lower end of the y-axis
            ymax: Upper end of the y-axis

        Returns:
            :class:`matplotlib.figure.Figure`: The figure containing the full
            plot including one or more curves and working points
        '''
        # remove any old WP legends
        for child in self.ax.get_children():
            if isinstance(child, mpl.legend.Legend):
                child.remove()

        self.ax.set_xlim(left=xmin, right=xmax)
        self.ax.set_ylim(bottom=ymin, top=ymax)
        self.ax.set_xlabel('false positive rate = 1 - specificity [%]')
        self.ax.set_ylabel('true positive rate = sensitivity [%]')

        legend = self.ax.legend(handles=self.get_legend_lines()[::-1],
                                labels = self.get_legend_labels()[::-1],
                                fontsize='small',
                                loc="lower right",
                                bbox_to_anchor=(1.0, 0.0), #(0.99, 0.01),
                                bbox_transform=self.ax.transAxes)

        # needed to get the following positions and transforms right
        self.fig.canvas.draw()

        # Adding the legend for the ROC curves removes the previously drawn
        # legend for the WP(s). Therefore the latter needs to be added back
        # as described in
        # https://riptutorial.com/matplotlib/example/32429/multiple-legends-on-the-same-axes
        if hasattr(self, 'legend_WP'):
            self.ax.add_artist(self.legend_WP)
            # get position of legend as described in
            # https://stackoverflow.com/q/28711376
            y1 = legend.legendPatch.get_bbox().transformed(
                self.ax.transAxes.inverted()).y1
            self.legend_WP.set_bbox_to_anchor((1.0, y1))

        return(self.fig)


    def pickle_save(self, filename: str = 'roc.pickle') -> None:
        '''
        Pickle this instance/object and save it to a file

        Based on https://stackoverflow.com/q/35649603 and
        https://stackoverflow.com/a/35667484

        Args:
            filename(str): The name of the file to which the object should be
                saved
        '''
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        print("Pickled ROCPlot to file '{}'".format(filename))


    @classmethod
    def pickle_load(cls, filename: str = 'roc.pickle') -> ROCPlot:
        '''
        Unpickle and load an instance/object from a file

        Based on https://stackoverflow.com/q/35649603 and
        https://stackoverflow.com/a/35667484

        Args:
            filename(str): The name of the file from which the object should be
                loaded
        
        Returns:
            ROC: The unpickled instance / object
        '''
        return(pickle.load(open(filename, 'rb')))



def plot_history(
    history: Union[dict, pd.DataFrame, tf.keras.callbacks.History],
    nrows: int = 1,
    ncols: int = 1,
    metrics: List[List[str]] = [['loss', 'val_loss']],
    names: List[List[str]] = [['training loss', 'validation loss']],
    axes: List[List[str]] = [['l', 'l']],
    colors: Union[str, List[List[str]]] = 'C0',
    zorders: Union[float, List[List[float]]] = 2.0,
    scaling_factors: List[List[float]] = [[1.0, 1.0]],
    formats: List[List[str]] = [['.3f', '.3f']],
    units: List[List[str]] = [['', '']],
    titles: List[str] = [None],
    y1labels: List[str] = ['loss'],
    y2labels: List[str] = ['accuracy'],
    legend_positions: List[str] = ['bl'],
    fontsize: float = None,
    figure_args: dict = {}
    ) -> mpl.figure.Figure:
    '''
    Plot the timeline of training metrics

    .. image:: ../images/loss-acc.png
       :width: 500px
       :align: center

    Args:
        history (:class:`dict`, :class:`pandas.DataFrame` or \
        :class:`tf.keras.callbacks.History`):
            *TensorFlow* history object returned by the model training or read
            from a ``*.csv`` file produced with the
            :class:`tf.keras.callbacks.CSVLogger` callback during training
        nrows (int): *Optional*. Number of rows of plots arranged in a grid
        ncols (int): *Optional*. Number of columns of plots arranged in a grid
        metrics ([[str]]): *Optional*. Names of the metrics in the history
            object that should be plotted
        names ([[str]]): *Optional*. How the plotted metrics should be named
            in the legends
        axes ([[str]]): *Optional*. ``l`` or ``r`` to describe on which y-axis
            a particular variable should be drawn
        colors (str / [[str]]): *Optional*. The line color(s). Can be either
            a single color string, in which case all lines will have this same
            color, or a list of lists of strings specifying the line colors for
            each metric.
        zorders (float / [[float]]): *Optional*. The zorder(s). Can be either
            a single float, in which case all zorders will have this same value,
            or a list of lists of floats specifying the zorders for each
            metric.
        scaling_factors ([[float]]): *Optional*. Scaling factors that multiply
            each metric
        formats ([[str]]): *Optional*. Format strings specifying how last value
            of each metric should be printed in the legends. Metrics are floats.
        units ([[str]]): *Optional*. Units that should be appended to the
            legend entry for each metric
        titles ([str]): *Optional*. List containing one title for each plot in
            the grid
        y1labels ([str]): *Optional*. List containing one label for the left
            axes for each plot in the grid
        y2labels ([str]): *Optional*. List containing one label for the right
            axes for each plot in the grid
        legend_positions ([str]): *Optional*. List containing a two-character
            string governing the legend position for each plot in the grid.
            Possible values are ``tl``, ``tc``, ``tr``, ``cl``, ``cc``, ``cr``,
            ``bl``, ``bc`` and ``br`` as implemented in
            :func:`spellbook.plotutils.legend_loc`
            and :func:`spellbook.plotutils.legend_bbox_to_anchor`.
        fontsize: *Optional*, baseline fontsize for all elements.
            This is probably the fontsize that ``medium`` corresponds to?
        figure_args: *Optional*, arguments for the creation of the
            :class:`matplotlib.figure.Figure` with
            :func:`matplotlib.pyplot.figure`

    Returns:
        :class:`matplotlib.figure.Figure`: The figure containing the plot or
        grid of plots
    '''

    # convert history to a dict of lists, including list of epochs
    if isinstance(history, pd.DataFrame):
        history = {col: history[col].values for col in history}
    if isinstance(history, tf.keras.callbacks.History):
        history.history['epoch'] = history.epoch
        history = history.history
    assert(isinstance(history, dict))
    assert('epoch' in history)

    if isinstance(colors, str):
        colors = np.full(shape = (len(metrics), len(metrics[0])),
                         fill_value = colors)
    if isinstance(zorders, float) or isinstance(zorders, int):
        zorders = np.full(shape = (len(metrics), len(metrics[0])),
                          fill_value = zorders)

    nGrid = len(metrics)
    assert(nGrid == len(names))
    assert(nGrid == len(axes))
    assert(nGrid == len(colors))
    assert(nGrid == len(zorders))
    assert(nGrid == len(scaling_factors))
    assert(nGrid == len(formats))
    assert(nGrid == len(units))
    assert(nGrid == len(titles))
    assert(nGrid == len(y1labels))
    assert(nGrid == len(y2labels))
    assert(nGrid == len(legend_positions))
    assert(nGrid <= nrows * ncols)

    if fontsize:
        tmp_fontsize = plt.rcParams['font.size']
        plt.rcParams['font.size'] = fontsize
    fig = plt.figure(**figure_args)
    grid = mpl.gridspec.GridSpec(nrows=nrows, ncols=ncols)

    for iGrid in range(nGrid):
        
        lastrow = (iGrid//ncols + 1 == nrows)

        lines = []
        labels = []
        ax = {}
        ax['l'] = plt.Subplot(fig, grid[iGrid])
        fig.add_subplot(ax['l'])
        if titles[iGrid]: ax['l'].set_title(titles[iGrid])
        ax['l'].get_xaxis().set_visible(lastrow)
        ax['l'].set_xlabel("epoch", fontsize="large")
        ax['l'].set_ylabel(y1labels[iGrid], fontsize="large")
        if 'r' in axes[iGrid]:
            ax['r'] = ax['l'].twinx() # second y-axis, share x-axis
            fig.add_subplot(ax['r'])
            ax['r'].set_ylabel(y2labels[iGrid], fontsize="large")

        for i in range(len(metrics[iGrid])):
            linestyle = '-' if 'val_' in metrics[iGrid][i] else '--'
            lines += ax[axes[iGrid][i]].plot(
                history['epoch'],
                # history[metric] is list, use numpy.array for scaling
                np.array(history[metrics[iGrid][i]]) \
                    * scaling_factors[iGrid][i],
                color=colors[iGrid][i],
                linestyle=linestyle,
                zorder = zorders[iGrid][i])
            # format specifier from variable: https://stackoverflow.com/a/32413139
            labels.append('{}: {:{format}}{}'.format(
                names[iGrid][i],
                history[metrics[iGrid][i]][-1] \
                    * scaling_factors[iGrid][i],
                units[iGrid][i],
                format = formats[iGrid][i]))

        ax['l'].xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
        ax['l'].set_ylim(bottom=0.0)
        if 'r' in axes[iGrid]:
            ax['r'].set_ylim(bottom=0.0)

        # bring ax['l] to the front so that lines in ax['r] are not drawn over
        # the legend, based on https://stackoverflow.com/a/33150705
        ax['l'].set_zorder(1)
        # make ax['l'] transparent so that it does not cover ax['r'] below
        # based on https://stackoverflow.com/a/33298097
        ax['l'].patch.set_alpha(0.0)
        # https://matplotlib.org/stable/tutorials/intermediate/legend_guide.html#legend-location
        # https://matplotlib.org/stable/tutorials/advanced/transforms_tutorial.html
        ax['l'].legend(handles = lines,
                       labels = labels,
                       loc = sb.plotutils.legend_loc(legend_positions[iGrid]),
                       bbox_to_anchor = sb.plotutils.legend_bbox_to_anchor(
                           legend_positions[iGrid]),
                       bbox_transform = ax['l'].transAxes)

    fig.tight_layout()
    if fontsize: plt.rcParams['font.size'] = tmp_fontsize
    return(fig)


def plot_history_binary(
    history: Union[dict, pd.DataFrame, tf.keras.callbacks.History],
    name_prefix: str = 'history-binary'
    ) -> None:
    '''
    Convenience function for plotting binary classification metrics

    .. list-table::
       :class: spellbook-gallery-scroll

       * - .. image:: ../images/loss-acc.png
              :height: 200px

         - .. image:: ../images/true-false-pos-neg-rates.png
              :height: 200px

         - .. image:: ../images/rec-prec.png
              :height: 200px

    The following plots are generated:

    - ``<name_prefix>-loss-acc.png``: Loss (binary crossentropy) and accuracy
    - ``<name_prefix>-pos-neg.png``: Numbers of true/false positives/negatives
    - ``<name_prefix>-rec-prec.png`` :term:`Recall/sensitivity<TPR>` and 
      :term:`precision`

    The metrics are defined as:

    - Recall = Sensitivity = True Positives / (True Positives + False Negatives)
    - Precision = True Positives / (True Positives + False Positives)

    In the plot containing the true/false positives/negatives, the left/primary
    and right/secondary y-axes are scaled relative to each other according to
    the ratio of the sizes of the training and validation datasets. Therefore,
    for a correctly working model and in the absence of significant
    overtraining, the training and validation curves should lie more or less
    on top of each other.

    Args:
        history (:class:`dict`, :class:`pandas.DataFrame` or \
        :class:`tf.keras.callbacks.History`):
            *TensorFlow* history object returned by the model training or read
            from a ``*.csv`` file produced with the
            :class:`tf.keras.callbacks.CSVLogger` callback during training
        name_prefix(str): Prefix to the filenames of the plots
    '''

    # convert history to a dict of lists, including list of epochs
    if isinstance(history, pd.DataFrame):
        history = {col: history[col].values for col in history}
    if isinstance(history, tf.keras.callbacks.History):
        history.history['epoch'] = history.epoch
        history = history.history
    assert(isinstance(history, dict))
    assert('epoch' in history)

    # This is a really clumsy way of getting the number of training and
    # validation datapoints. When I tried it, the tf.keras.metrics.Sum threw an
    # exception, so this is a workaround.
    def count_datapoints(prefix=''):
        tp = history[prefix+'tp'][-1]
        tn = history[prefix+'tn'][-1]
        fp = history[prefix+'fp'][-1]
        fn = history[prefix+'fn'][-1]
        return(tp + tn + fp + fn)
    n_train = count_datapoints()
    n_test = count_datapoints('val_')
    ratio = n_train / n_test

    sb.plot.save(
        plot_history(
            history,
            nrows = 1,
            ncols = 1,
            metrics = [['binary_crossentropy', 'val_binary_crossentropy',
                        'binary_accuracy', 'val_binary_accuracy']],
            names = [['training loss', 'validation loss',
                    'training accuracy', 'validation accuracy']],
            axes = [['l', 'l', 'r', 'r']],
            colors = [['C0', 'C0', 'C1', 'C1']],
            scaling_factors = [[1.0, 1.0, 100.0, 100.0]],
            formats = [['.3f', '.3f', '.1f', '.1f']],
            units = [['', '', '%', '%']],
            titles = [None],
            y1labels = ['loss: binary crossentropy'],
            y2labels = ['accuracy [%]'],
            legend_positions = ['cr']
        ),
        name_prefix + '-loss-acc.png'
    )

    fig = plot_history(
        history,
        nrows = 2,
        ncols = 2,
        metrics = [
            ['tn', 'val_tn'], ['fp', 'val_fp'],
            ['fn', 'val_fn'], ['tp', 'val_tp']],
        names = [
            ['training', 'validation'], ['training', 'validation'],
            ['training', 'validation'], ['training', 'validation']
        ],
        axes = [['l', 'r'], ['l', 'r'], ['l', 'r'], ['l', 'r']],
        colors = [['C1', 'C0'], ['C1', 'C0'], ['C1', 'C0'], ['C1', 'C0']],
        scaling_factors = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
        formats = [['.0f', '.0f'], ['.0f', '.0f'], ['.0f', '.0f'], ['.0f', '.0f']],
        units = [['', ''], ['', ''], ['', ''], ['', '']],
        titles = ['True Negatives', 'False Positives', 'False Negatives', 'True Positives'],
        y1labels = ['training', '', 'training', ''],
        y2labels = ['', 'validation', '', 'validation'],
        legend_positions = ['bl', 'tr', 'tr', 'bl'],
        figure_args = dict(figsize = (9.6, 7.2))
    )
    # scale the ranges of the left and right y-axes so that they properly
    # reflect the different sizes of the training and test datasets
    for iGrid in range(4):
        y1 = fig.get_axes()[2*iGrid].get_ylim()[1]
        y2 = fig.get_axes()[2*iGrid+1].get_ylim()[1]
        fig.get_axes()[2*iGrid].set_ylim(top = max(y1, y2 * ratio))
        fig.get_axes()[2*iGrid+1].set_ylim(top = max(y2, y1 / ratio))
    sb.plot.save(fig, name_prefix + '-pos-neg.png')

    sb.plot.save(
        plot_history(
            history,
            nrows = 1,
            ncols = 1,
            metrics = [['recall', 'val_recall', 'precision', 'val_precision']],
            names = [['training recall/sensitivity', 'validation recall/sensitivity',
                      'training precision', 'validation precision']],
            axes = [['l', 'l', 'l', 'l']],
            colors = [['C0', 'C0', 'C1', 'C1']],
            scaling_factors = [[100.0, 100.0, 100.0, 100.0]],
            formats = [['.1f', '.1f', '.1f', '.1f']],
            units = [['%', '%', '%', '%']],
            titles = [''],
            y1labels = ['value [%]'],
            y2labels = [None],
            legend_positions = ['br']
        ),
        name_prefix + '-rec-prec.png'
    )


def get_binary_labels(predictions: Union[np.ndarray, tf.Tensor],
                      threshold: float = 0.5) -> np.ndarray:
    '''
    Apply threshold to calculate the labels from the predictions

    It is necessary to apply the threshold and unequivocally assign each
    datapoint to a category for the calculation of the confusion matrix to
    work correctly in *TensorFlow*. When given the sigmoid-activated classifier
    output, the confusion matrix calculation will otherwise floor() the
    predictions to zero and associate all datapoints with the first/negative
    class.

    Args:
        predictions: The sigmoid-activated classifier outputs, one value for
            each datapoint
        threshold: Datapoints whose prediction is below the threshold are
            associated to the first/negative class, datapoints whose prediction
            is above the threshold to the second/positive class
    
    Returns:
        Array of predicted labels
    
    Examples:

    .. doctest::

        >>> import numpy as np
        >>> import spellbook as sb

        >>> predictions = np.arange(0.0, 1.0, 0.2)
        >>> print(predictions)
        [0.  0.2 0.4 0.6 0.8]

        >>> # default threshold of 0.5
        >>> predicted_labels = sb.train.get_binary_labels(predictions)
        >>> print(predicted_labels)
        [False False False  True  True]

        >>> # custom threshold of 0.7
        >>> predicted_labels = sb.train.get_binary_labels(predictions, 0.7)
        >>> print(predicted_labels)
        [False False False False  True]

        >>> # type(predictions)
        >>> predicted_labels = sb.train.get_binary_labels((0.1, 0.5, 0.9))
        Traceback (most recent call last):
           ...
        TypeError: Argument 'predictions' must be of type 'numpy.ndarray' or 'tensorflow.Tensor'
    '''
    # based on https://stackoverflow.com/a/41672309
    if isinstance(predictions, np.ndarray):
        return(np.array(predictions > threshold))
    elif isinstance(predictions, tf.Tensor):
        return(np.array(predictions.numpy() > threshold))
    else:
        raise TypeError("Argument 'predictions' must be of "
                        "type 'numpy.ndarray' or 'tensorflow.Tensor'")


def plot_calibration_curve(
    labels: Union[np.ndarray, tf.Tensor],
    predictions: Union[np.ndarray, tf.Tensor],
    n_bins: int = 10,
    histogram_args: dict = {}):
    '''
    .. todo:: write docstring for spellbook.train.plot_calibration_curve

    .. image:: /images/calibration.png
       :align: center
       :width: 400px
    '''

    # calculate calibration curve
    prob_true, prob_pred = sklearn.calibration.calibration_curve(
        y_true=labels, y_prob=predictions, n_bins=n_bins)

    # prepare the figure
    fig = mpl.pyplot.figure(figsize=(5.0,6.7), tight_layout=True)
    # https://matplotlib.org/stable/tutorials/intermediate/gridspec.html
    grid = mpl.gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[0.75, 0.25])

    # plot the calibration curve
    ax_calibration = mpl.pyplot.Subplot(fig, grid[0])
    fig.add_subplot(ax_calibration)
    ax_calibration.set_aspect('equal')
    ax_calibration.plot([0.0, 1.0], [0.0, 1.0], '--', color='grey')
    ax_calibration.plot(prob_pred, prob_true, linestyle='-', marker='o')
    ax_calibration.set_xlabel('mean predicted value')
    ax_calibration.set_ylabel('fraction of positives')

    # plot the histogram of the predictions
    sb.plot1D.histogram(data=predictions, fig=fig, grid=grid, gridindex=1,
        xlabel='prediction', show_decorations=False, show_stats=False,
        **histogram_args)

    return fig
