# Functions for evaluating model performance

import seaborn as sns

def plot_residuals(ground_truth, predictions, bins=50):
    '''
    Plots predictions against ground truth
    '''
    
    residuals = ground_truth - predictions
    
    (sns.jointplot(x=predictions, y=residuals, marginal_kws=dict(bins=bins))
     .set_axis_labels('predictions', 'residuals'))
