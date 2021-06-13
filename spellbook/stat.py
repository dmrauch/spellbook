'''
Statistics functions
'''

import math
import numpy as np
from scipy import stats

import spellbook as sb


def describe(data, CL_mean=95):

  s = stats.describe(data)
  n = s.nobs
  mean = s.mean
  std = math.sqrt(s.variance)  # standard deviation
  sem = stats.sem(data) # standard error of the mean
  t = math.fabs(stats.t(df=n-1).ppf((1-CL_mean/100.0)/2.0))
  CI_mean = t * sem

  med = np.median(data)
  quantiles = np.quantile(data, [0.0, 0.25, 0.5, 0.75, 1.0])

  return({
    'count': n,
    'mean': mean,
    'std': std,
    'sem': sem,
    'CL_mean': CL_mean,
    'CI_mean': CI_mean,
    'median': med,
    'min': quantiles[0],
    '25%': quantiles[1],
    '75%': quantiles[3],
    'max': quantiles[4]
  })


def describe_text(data):

  stats = describe(data)
  labels = ['count', 'mean', 'std', 'min', '25%', 'median', '75%', 'max']
  values = ['{}'.format(stats['count']),
            '{:.2f}'.format(stats['mean']),
            '{:.2f}'.format(stats['std']),
            '{:.2f}'.format(stats['min']),
            '{:.2f}'.format(stats['25%']),
            '{:.2f}'.format(stats['median']),
            '{:.2f}'.format(stats['75%']),
            '{:.2f}'.format(stats['max'])]
  return(sb.plotutils.valuebox_text(labels, values))
