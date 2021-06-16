import matplotlib as mpl
import pandas as pd

import spellbook as sb

import helpers

# data loading and cleaning
data, vars, target, features = helpers.load_data()

# inplace convert string category labels to numerical indices
categories = sb.input.encode_categories(data)

# oversampling (including shuffling of the data)
data = sb.input.oversample(data, target)

# use new numerical columns for the features
for var in categories:
    if var == target:
        target = target + '_codes'
    else:
        features[features.index(var)] = var + '_codes'

fig = sb.plot.parallel_coordinates(data.sample(frac=1.0).iloc[:500],
    features, target, categories)
sb.plot.save(fig, 'parallel-coordinates.png')
