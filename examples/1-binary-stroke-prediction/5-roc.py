import numpy as np

import spellbook as sb
from spellbook.train import ROCPlot


# prefix = ''
prefix = 'runs/'

# roc_naive10 = ROCPlot.pickle_load(prefix+'naive-e10-roc.pickle')
roc_naive100 = ROCPlot.pickle_load(prefix+'naive-e100-roc.pickle')
# roc_oversampling10 = ROCPlot.pickle_load(prefix+'oversampling-e10-roc.pickle')
roc_oversampling2000 = ROCPlot.pickle_load(prefix+'oversampling-e2000-roc.pickle')
# roc_norm10 = ROCPlot.pickle_load(prefix+'oversampling-normalised-e10-roc.pickle')
roc_norm2000 = ROCPlot.pickle_load(prefix+'oversampling-normalised-e2000-roc.pickle')


# # naive versus oversampling vs normalised for 10 training epochs

# roc = ROCPlot()
# roc += roc_naive10
# roc += roc_oversampling10
# roc += roc_norm10
# WPs = []
# WPs.append(roc.get_WP('naive / 10 epochs (validation)', threshold=0.5))
# WPs.append(roc.get_WP('oversampling / 10 epochs (validation)', threshold=0.5))
# WPs.append(roc.get_WP('oversampling normalised / 10 epochs (validation)', threshold=0.5))
# roc.draw_WP(WPs, linestyle='-', linecolor=['C1', 'C0', 'black'])
# roc.curves['naive / 10 epochs (training)']['line'].set_color('C1')
# roc.curves['naive / 10 epochs (validation)']['line'].set_color('C1')
# roc.curves['oversampling normalised / 10 epochs (training)']['line'].set_color('black')
# roc.curves['oversampling normalised / 10 epochs (validation)']['line'].set_color('black')
# sb.plot.save(roc.plot(), prefix+'roc-10-naive-oversampling-normalised.png')


# naive versus oversampling vs normalised for 2000 training epochs

roc = ROCPlot()
roc += roc_naive100
WP = roc.get_WP('naive / 100 epochs (validation)', threshold=0.5)
roc.draw_WP(WP, linestyle='-', linecolor='C1')
roc.curves['naive / 100 epochs (training)']['line'].set_color('C1')
roc.curves['naive / 100 epochs (validation)']['line'].set_color('C1')
sb.plot.save(roc.plot(), prefix+'roc-100-naive.png')

roc += roc_oversampling2000
WP = roc.get_WP('oversampling / 2000 epochs (validation)', threshold=0.5)
roc.draw_WP(WP, linestyle='-', linecolor='C0')
sb.plot.save(roc.plot(), prefix+'roc-2000-naive-oversampling.png')

roc += roc_norm2000
WP = roc.get_WP('oversampling normalised / 2000 epochs (validation)', threshold=0.5)
roc.draw_WP(WP, linestyle='-', linecolor='black')
roc.curves['oversampling normalised / 2000 epochs (training)']['line'].set_color('black')
roc.curves['oversampling normalised / 2000 epochs (validation)']['line'].set_color('black')
sb.plot.save(roc.plot(), prefix+'roc-2000-naive-oversampling-normalised.png')
