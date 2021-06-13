import spellbook as sb

import helpers

# data loading and cleaning
data, vars, target, features = helpers.load_data()


# ------------------------------------------------------------------------------
# DATA VISUALISATION
# ------------------------------------------------------------------------------

# plot individual distributions

# stroke
fig = sb.plot.plot_1D(data=data, x='stroke', fontsize=14.0)
sb.plot.save(fig, filename='stroke.png')

# average glucose level
sb.plot.save(
    sb.plot.plot_1D(
        data = data,
        x = 'avg_glucose_level',
        xlabel = 'Average Glucose Level',
        statsbox_args = {
            'text_args': {'backgroundcolor': 'white'}
        }
    ),
    filename='avg_glucose_level.png'
)

# plot all target and feature distributions
sb.plot.save(
    sb.plot.plot_grid_1D(
        nrows=3, ncols=4, data=data, target=target, features=features,
        stats=False, fontsize=11.0
    ),
    filename='variables.png'
)

# plot a subset of the feature variables
sb.plot.save(
    sb.plot.plot_grid_1D(
        nrows=2, ncols=3, data=data,
        features=['age', 'hypertension', 'heart_disease',
                  'bmi', 'avg_glucose_level', 'smoking_status'],
        stats=False, fontsize=11.0
    ),
    filename='variables-health.png'
)

# plot the correlations between individual features and the target
sb.plot.save( # default: no descriptive statistics
    sb.plot.plot_2D(data=data, x='age', y=target, fontsize=14.0),
    filename='age-corr.png'
)
sb.plot.save( # boxes with descriptive statistics included
    sb.plot.plot_2D(
        data=data, x='age', y=target, fontsize=11.0,
        cathist_args = {
            'histogram_args': [
                dict(show_stats=True, statsbox_args={'alignment': 'bl'}),
                dict(
                    show_stats = True,
                    statsbox_args = {
                        'y': 0.96,
                        'text_args': {
                            # RGBA white with 50% alpha/opacity
                            'backgroundcolor': (1.0, 1.0, 1.0, 0.5)
                        }
                    }
                )
            ]
        }
    ),
    filename='age-corr-stats.png'
)


# plot all correlations between the features and the target

# absolute values
fig = sb.plot.plot_grid_2D(nrows=2, ncols=5, data=data, xs=features, ys=target)
sb.plot2D.heatmap_set_annotations_fontsize(
    ax=fig.get_axes()[7], fontsize='x-small')
sb.plot2D.heatmap_set_annotations_fontsize(
    ax=fig.get_axes()[15], fontsize='small')
sb.plot.save(fig, 'corrs-absolute.png')

# relative values
fig = sb.plot.plot_grid_2D(
    nrows=2, ncols=5, data=data, xs=features, ys=target, relative='true')
sb.plot2D.heatmap_set_annotations_fontsize(
    ax=fig.get_axes()[7], fontsize='x-small')
sb.plot2D.heatmap_set_annotations_fontsize(
    ax=fig.get_axes()[15], fontsize='small')
sb.plot.save(fig, 'corrs-relative.png')


# correlations between the features

# specific correlations
sb.plot.save(sb.plot.plot_2D(data=data, x='age', y='ever_married'),
    filename='corr-married-age.png')
sb.plot.save(sb.plot.plot_2D(data=data, x='age', y='work_type'),
    filename='corr-work-age.png')

# pairplots

# 3x5 pairplot
vars = ['ever_married', 'age', 'hypertension', 'heart_disease', 'bmi', 'stroke']
pairplot = sb.plot.pairplot(data, xs=vars[:5], ys=vars[1:4])
sb.plot.save(pairplot, 'pairplot-3x5.png')

# 5x5 pairplot
vars = ['gender', 'age', 'hypertension', 'bmi', 'stroke']
pairplot = sb.plot.pairplot(data, xs=vars)
sb.plot.save(pairplot, 'pairplot-5x5.png')

# full pairplot of all features
# sb.plot.save(
#     sb.plot.pairplot(data, xs=features),
#     filename='pairplot-features.png', dpi=100)
