# Functions for evaluating model performance

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_style('whitegrid')

def plot_residuals(ground_truth, predictions, bins=50):
    '''
    Plots residuals against the predictions given ground_truth, including histograms.
    '''
    
    residuals = ground_truth - predictions
    
    (sns.jointplot(x=predictions, y=residuals, marginal_kws=dict(bins=bins))
     .set_axis_labels('predictions', 'residuals'))
    
# 
def MRE(ground_truth, predictions, target_log1p_transform=False, plot=False):
    '''
    Calculates the median relative error / median absolute percentage error (MAPE) of predictions given ground_truth.
    Setting `plot = True` also plots the distribution of errors.
    Setting `target_log1p_transform = True` assumes that both `ground_truth` as well as `predictions` have been `log1p` transformed.
    '''
    
    if target_log1p_transform:
        ground_truth = np.expm1(ground_truth)
        predictions = np.expm1(predictions)
    
    diff = np.abs(predictions / ground_truth - 1)
    
    if plot:
        plt.figure(figsize=(10,5));
        sns.distplot(100 * (predictions / ground_truth - 1), rug=True, hist=True);
        plt.xlabel('relative error [%]');
    
    return np.median(diff)
