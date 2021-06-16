'''
Functions for model inspection
'''

from __future__ import annotations   # for type hinting the enclosing class
                                     # https://stackoverflow.com/a/33533514
import matplotlib as mpl
import numpy as np
import pandas as pd
import scipy
import sys
import tensorflow as tf

modules = {}
try:
    import tensorflow_decision_forests as tfdf
    modules['tfdf'] = True
except:
    modules['tfdf'] = False

from typing import Dict
from typing import List
from typing import Union

import spellbook as sb


class PermutationImportance:
    '''
    Feature importance from permutation

    This implementation follows the *Permutation Feature Importance*
    algorithm in *scikit-learn*

    - presented in the *scikit-learn* `User Guide
      <https://scikit-learn.org/stable/modules/permutation_importance.html>`_
    - implemented in :func:`sklearn.inspection.permutation_importance`

    but goes further in that it provides a mechanism for permuting
    clusters of multiple features simultaneously. This allows to
    estimate the permutation importance when some of the feature
    variables are correlated.

    Args:
        data (:class:`pandas.DataFrame`): The dataset
        features: The names of the feature variables
        target: The name of the target variable
        model (:class:`tf.keras.Model`): The predictor/classifier/regressor
        metrics ([:class:`tf.keras.metrics.Metric`]):
            The metrics to evaluate
        n_repeats: How often each feature (cluster) is permuted
        feature_clusters: *Optional*. Dictionary with cluster names as keys
            and lists of features as the values. Each list contains the
            features that are grouped together as one cluster and permuted
            simultaneously.
        tfdf: *Optional*. Whether or not the model is one of the models
            in ``tensorflow_decision_forests``
    
    Attributes:
        baseline (:class:`dict`\[:class:`str`, :class:`float`\]):
            Dictionary containing the nominal metrics, i.e. without permutation.
            The keys are the names of the metrics and the values are the values
            of the metrics.
        results (:class:`list`\[:class:`dict`\]): For each feature or feature
            cluster, a dictionary is added to the list. Each dictionary has
            the following keys and associated values:

            - ``feature``: The name of the feature or the feature cluster
            - ``results``: A list containing a dictionary for each permutation.
              Each dictionary contains the names and values of the metrics
              calculated in that permutation
            - ``mean``: A dictionary containing the means of the results,
              with one entry for each metric
            - ``std``: A dictionary containing the standard deviations of the
              results, with one entry for each metric
            - ``mean_rel_diff``: A dictionary containing the relative
              differences between the mean and the nominal, with one entry
              for each metric

        tfdf (bool): Whether or not the model is one of the models in
            ``tensorflow_decision_forests``

    See also:

        - *scikit-learn* example:
          `Permutation Importance with Multicollinear or Correlated Features
          <https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html>`_
    '''


    def __init__(self,
        data: pd.DataFrame,
        features,
        target,
        model,
        metrics: Union[tf.keras.metrics.Metric, List[tf.keras.metrics.Metric]],
        n_repeats: int = 10,
        feature_clusters: Dict[str, List[str]] = None,
        tfdf: bool = False) -> PermutationImportance:

        self.tfdf = tfdf
        if tfdf and not modules['tfdf']:
            raise ModuleNotFoundError(
                "No module named 'tensorflow_decision_forests")
        
        if not isinstance(metrics, list): metrics = [metrics]
        assert isinstance(n_repeats, int) and n_repeats > 0

        # get the nominal model performance without feature permutation
        self.baseline = self._permute_feature(
            data, features, target, model, metrics)

        # calculate permutation runs
        if feature_clusters:
            runs = {}
            buffer = features.copy()
            for cluster_name, cluster_features in feature_clusters.items():
                runs[cluster_name] = cluster_features
                for feature in cluster_features:
                    # each feature in the cluster must exist
                    assert feature in features
                    # each feature can only be associated to one cluster
                    assert feature in buffer
                    buffer.remove(feature)
            for feature in buffer:
                runs[feature] = [feature]
        else:
            runs = {feature: [feature] for feature in features}

        # feature permutation loop
        self.results = []
        print('Calculating permutation importance:')
        for irun, (run_name, run_vars) in enumerate(runs.items()):

            # prepare the data structures
            result = {}
            result['feature'] = run_name # var
            result['results'] = []

            # get the results
            for irep in range(n_repeats):
                # https://stackoverflow.com/a/52590238
                # if irep > 0: sys.stdout.write('\x1b[1A') # cursor up one line
                sys.stdout.write('\r') # carriage return
                sys.stdout.write('\x1b[2K') # delete last line
                sys.stdout.write("- feature/cluster '{}' - {}/{} - {:.1f}% "\
                    .format(run_name, irep+1, n_repeats,
                        (irun*n_repeats+irep)/(len(runs)*n_repeats)*100.0))
                sys.stdout.flush()

                result['results'].append(
                    self._permute_feature(
                        data, features, target, model, metrics, run_vars))

            # calculate the mean and standard deviation for each metric
            result['mean'] = {}
            result['std'] = {}
            for metric in metrics:
                values = [res[metric.name] for res in result['results']]
                stats = scipy.stats.describe(values)
                result['mean'][metric.name] = stats.mean
                result['std'][metric.name] = np.sqrt(stats.variance)
            
            # calculate the relative difference of the mean to the baseline
            # for each metric
            result['mean_rel_diff'] = {}
            for metric in metrics:
                result['mean_rel_diff'][metric.name] \
                    = (result['mean'][metric.name] - self.baseline[metric.name]) \
                        / self.baseline[metric.name]

            self.results.append(result)

            # https://stackoverflow.com/a/52590238
            # sys.stdout.write('\x1b[1A') # cursor up one line
            sys.stdout.write('\r') # carriage return
            sys.stdout.write('\x1b[2K') # delete last line
            print("- feature/cluster '{}' - DONE - {:.1f}%".format(run_name,
                ((irun+1)*n_repeats)/(len(runs)*n_repeats)*100.0))


    def _permute_feature(self,
        data: pd.DataFrame,
        features: List[str],
        target: str,
        model: tf.keras.Model,
        metrics: List[tf.keras.metrics.Metric],
        permute: Union[str, List[str]] = None) -> Dict[str, float]:
        '''
        Calculate model metrics for dataset with one or more features permuted

        The feature(s) given by **permute** are permuted. This may be a single
        feature or a list of features, in which case all features in the list
        are permuted in an uncorrelated manner, i.e. simultaneously and
        independently of each other.

        Args:
            data (:class:`pandas.DataFrame`): The dataset
            features: The names of the feature variables
            target: The name of the target variable
            model (:class:`tf.keras.Model`): The predictor/classifier/regressor
            metrics ([:class:`tf.keras.metrics.Metric`]):
                The metrics to evaluate
            permute: *Optional*. The name(s) of the variable(s) to permute
                (independently)

        Returns:
            A dictionary - the keys are the names of the metrics and the values
            are the values of the metrics
        '''

        if permute:
            # either str or List[str]
            assert isinstance(permute, str) or isinstance(permute, list)
            
            # promote to List[str]
            if isinstance(permute, str): permute = [permute]

        local_data = data[features + [target]].copy()
        if permute:
            for pf in permute:
                assert isinstance(pf, str)
                assert pf in features

                # permute one single column: https://stackoverflow.com/a/54014039
                local_data[pf] = np.random.default_rng().permutation(
                    local_data[pf].values)
        
        if self.tfdf and modules['tfdf']:
            dataset = tfdf.keras.pd_dataframe_to_tf_dataset(
                local_data, label=target)
            xs, ys = zip(*dataset.unbatch())
            predictions = model.predict(dataset)

        else:
            # model is neural network

            xs = local_data[features].values
            ys = local_data[target].values

            # obtain the predictions of the model
            predictions = model.predict(xs) # when fed np.ndarrays returns np.ndarray
            # ys_pred = sb.train.get_binary_labels(predictions)

        # calculate the metrics
        result = {}
        for metric in metrics:
            metric.reset_state()
            metric.update_state(ys, predictions)
            result[metric.name] = metric.result().numpy()

        return result


    def plot(self,
        metric_name,
        xmin: float = None,
        xmax: float = None,
        ascending: bool = True,
        annotations_alignment: str = 'left',
        show_std: bool = False,
        show_rel_diffs: bool = True,
        rainbow: bool = False
        ) -> mpl.figure.Figure:
        '''
        Plot the permutation importance of features / feature clusters

        .. list-table::
           :class: spellbook-gallery-scroll
        
           * - .. image:: /images/permutation-importance.png
                  :width: 400px
            
             - .. image:: /images/permutation-importance-rainbow.png
                  :width: 400px
        
        Args:
            metric_name: The name of the metric to be plotted
            xmin: *Optional*. The lower end of the x-axis
            xmax: *Optional*. The upper end of the x-axis
            ascending: *Optional*. Order from the top to the bottom of the plot:

              - ``True``: Ascending from smaller to larger values
              - ``False``: Descending from larger to smaller values

            annotations_alignment: *Optional*. Whether the annotations
                indicating the mean (and possibly the standard deviation)
                as well as the relative difference to the nominal metric
                should be printed to the ``left`` or the ``right`` of the
                markers.
            show_std: *Optional*. Whether or not the standard deviations
                of the metrics for the permuted features should be included
                in the annotations.
            show_rel_diffs: *Optional*. Whether or not the relative
                differences between the mean of the metrics for the permuted
                features and the nominal metric shown be included in the
                annotations.
            rainbow: *Optional*. Whether or not the horizontal bars between
                the means of the metrics for the permuted features and the
                nominal metric should cycle through the colour palette.

        Returns:
            The figure containing the ranking of the features according to
            their permutation importance
        '''

        assert annotations_alignment in ['left', 'right']
        space_before = '   ' if annotations_alignment == 'right' else ''
        space_after = '   ' if annotations_alignment == 'left' else ''
        annotations_alignment \
            = 'left' if annotations_alignment=='right' else 'right'

        self.results.sort(
            key = lambda elem: elem['mean'][metric_name],
            reverse = ascending)

        fig, ax = mpl.pyplot.subplots(
            figsize=(6.0, 1.0+0.5*(len(self.results)+1)),
            tight_layout=True)

        yticklabels = []
        for i, result in enumerate(self.results):

            # bar from the mean to the baseline
            ax.errorbar(
                x = 0.5*(result['mean'][metric_name]+self.baseline[metric_name]),
                y = i,
                xerr = 0.5*np.fabs(result['mean'][metric_name]-self.baseline[metric_name]),
                linewidth = 7.5,
                color = None if rainbow else '#aacbe2' # '#1f77b460'
            )
            
            # mean value and standard deviation
            ax.errorbar(
                result['mean'][metric_name], i,
                xerr=result['std'][metric_name], yerr=0.5,
                color='C0', linewidth=4.0, zorder=2.1)
            
            # annotations with mean values and standard deviations
            if show_std:
                annotation = '{}{:.3f} ± {:.3f}{}'.format(
                    space_before,
                    result['mean'][metric_name],
                    result['std'][metric_name],
                    space_after)
            else:
                annotation = '{}{:.3f}{}'.format(
                    space_before,
                    result['mean'][metric_name],
                    space_after)
            ax.annotate(
                annotation,
                xy = (result['mean'][metric_name], i),
                xytext = (result['mean'][metric_name], i+0.26),
                horizontalalignment = annotations_alignment,
                verticalalignment = 'center',
                color = 'grey', fontsize = 'small', fontweight = 'semibold')
            
            # annotations with relative differences
            if show_rel_diffs:
                ax.annotate(
                    '{}{:+.1f}%{}'.format(
                        space_before,
                        100.0*result['mean_rel_diff'][metric_name],
                        space_after),
                    xy = (result['mean'][metric_name], i),
                    xytext = (result['mean'][metric_name], i-0.30),
                    horizontalalignment = annotations_alignment,
                    verticalalignment = 'center',
                    color = 'grey', fontsize = 'x-small') # fontstyle = 'italic'

            yticklabels.append(result['feature'])
        ax.set_ylim(top=len(self.results)+0.5)

        # transformation from data to axes coordinates to determine ymax for vlines
        # based on https://stackoverflow.com/q/62004022
        p = ax.transData.transform((self.baseline[metric_name], len(self.results)))
        xb, yb = ax.transAxes.inverted().transform(p)
        ax.text(
            x = 0.5, y = yb,
            s = 'nominal {}: {:.3f}'.format(metric_name, self.baseline[metric_name]),
            color='C1', transform=ax.transAxes, horizontalalignment='center')
        ax.axvline(self.baseline[metric_name], color='C1', linewidth=4.0)

        ax.set_yticks(np.arange(len(self.results)))
        ax.set_yticklabels(yticklabels)
        ax.set_xlabel(metric_name)
        if xmin: ax.set_xlim(left=xmin)
        if xmax: ax.set_xlim(right=xmax)
        ax.grid(which='major', axis='y', zorder=0.0)

        return fig
