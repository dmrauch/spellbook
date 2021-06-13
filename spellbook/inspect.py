'''
Functions for model inspection
'''

import matplotlib as mpl
import numpy as np
import pandas as pd
import scipy
import tensorflow as tf

from typing import Dict
from typing import List
from typing import Union

import spellbook as sb


class PermutationImportance:


    def __init__(self,
        data, features, target,
        model, metrics,
        n_repeats=10,
        feature_clusters: Dict[str, List[str]] = None):
        '''
        .. todo:: write docstring for spellbook.inspect.permutation_importance

        See also:
            This implementation follows the *Permutation Feature Importance*
            algorithm in *scikit-learn*

            - presented in
              https://scikit-learn.org/stable/modules/permutation_importance.html
            - implemented in
              https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html

            :Further reading:

            - Permutation Importance with Multicollinear or Correlated Features:
              https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html

        '''

        # get the nominal model performance without feature permutation
        self.baseline = self.permute_feature(
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
        # for var in features:
        for run_name, run_vars in runs.items():

            # prepare the data structures
            result = {}
            result['feature'] = run_name # var
            result['results'] = []

            # get the results
            for i in range(n_repeats):
                result['results'].append(
                    self.permute_feature(
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


    def permute_feature(self,
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
            metrics ([:class:`tf.keras.metrics.Metric`]): The metrics to evaluate
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

        local_data = data.copy()
        if permute:
            for pf in permute:
                assert isinstance(pf, str)
                assert pf in features

                # permute one single column: https://stackoverflow.com/a/54014039
                local_data[pf] = np.random.default_rng().permutation(
                    local_data[pf].values)

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
        show_rel_diffs: bool = True,
        rainbow: bool = False):
        '''
        Plot the feature permutation importance

        This function plots the feature permutation importance calculated
        with :func:`spellbook.inspect.permutation_importance`

        .. todo:: write docstring for spellbook.inspect.plot_permutation_importance

        .. list-table::
           :class: spellbook-gallery-scroll
        
           * - .. image:: /images/permutation-importance.png
                  :width: 400px
            
             - .. image:: /images/permutation-importance-rainbow.png
                  :width: 400px
        
        Args:
            ascending: *Optional*. Order from the top to the bottom of the plot:

            - ``True``: Ascending from smaller to larger values
            - ``False``: Descending from larger to smaller values

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
                color='C0', linewidth=4.0)
            
            # annotations with mean values
            ax.annotate(
                '{}{:.3f}{}'.format(
                    space_before,
                    result['mean'][metric_name],
                    space_after),
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
        ax.axvline(self.baseline[metric_name], color='C1', linewidth=3.0)

        ax.set_yticks(np.arange(len(self.results)))
        ax.set_yticklabels(yticklabels)
        ax.set_xlabel(metric_name)
        if xmin: ax.set_xlim(left=xmin)
        if xmax: ax.set_xlim(right=xmax)
        ax.grid(which='major', axis='y', zorder=0.0)

        return fig
