import spellbook as sb

# load the pickled ROC curves of the different models
roc_network = sb.train.ROCPlot.pickle_load(
    '../1-binary-stroke-prediction/oversampling-normalised-e2000-roc.pickle')
roc_forest = sb.train.ROCPlot.pickle_load('random-forest-roc.pickle')
roc_trees = sb.train.ROCPlot.pickle_load('gradient-trees-roc.pickle')

# add and style the ROC curves
roc = sb.train.ROCPlot()
roc += roc_network
roc.curves['oversampling normalised / 2000 epochs (validation)']['line'].set_color('black')
roc.curves['oversampling normalised / 2000 epochs (training)']['line'].set_color('black')
roc += roc_trees
roc.curves['gradient trees (validation)']['line'].set_color('C1')
roc.curves['gradient trees (training)']['line'].set_color('C1')
roc += roc_forest

# calculate and draw the working points with 100% true positive rate
WPs = []
WPs.append(roc.get_WP(
    'oversampling normalised / 2000 epochs (validation)', TPR=1.0))
WPs.append(roc.get_WP(
    'gradient trees (validation)', TPR=1.0))
WPs.append(roc.get_WP(
    'random forest (validation)', TPR=1.0))
roc.draw_WP(WPs, linecolor=['black', 'C1', 'C0'])

# save the plot
sb.plot.save(roc.plot(xmin=-0.2, xmax=11.0, ymin=50.0), 'roc.png')
