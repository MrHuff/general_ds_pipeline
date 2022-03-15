import matplotlib.pyplot as plt
from chr.methods import CHR
import pickle
import dill
import numpy as np

def plot_func(x, y, quantiles=None, quantile_labels=None, max_show=5000,
              shade_color="", method_name="", title="", filename=None, save_figures=False):

    """ Scatter plot of (x,y) points along with the constructed prediction interval

    Parameters
    ----------
    x : numpy array, corresponding to the feature of each of the n samples
    y : numpy array, target response variable (length n)
    quantiles : numpy array, the estimated prediction. It may be the conditional mean,
                or low and high conditional quantiles.
    shade_color : string, desired color of the prediciton interval
    method_name : string, name of the method
    title : string, the title of the figure
    filename : sting, name of the file to save the figure
    save_figures : boolean, save the figure (True) or not (False)

    """

    x_ = x[:max_show]
    y_ = y[:max_show]
    if quantiles is not None:
        quantiles = quantiles[:max_show]

    fig = plt.figure()
    inds = np.argsort(np.squeeze(x_))
    plt.plot(x_[inds], y_[inds], 'k.', alpha=.2, markersize=10, fillstyle='none')

    if quantiles is not None:
        num_quantiles = quantiles.shape[1]
    else:
        num_quantiles = 0

    if quantile_labels is None:
        pred_labels = ["NA"] * num_quantiles
    for k in range(num_quantiles):
        label_txt = 'Quantile {q}'.format(q=quantile_labels[k])
        plt.plot(x_[inds], quantiles[inds ,k], '-', lw=2, alpha=0.75, label=label_txt)

    # plt.ylim([-2, 20])
    plt.xlabel('$X$')
    plt.ylabel('$Y$')
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.title(title)
    if save_figures and (filename is not None):
        plt.savefig(filename, bbox_inches='tight', dpi=300)

    plt.show()

if __name__ == '__main__':
    data_path = 'housing'
    job_name = 'test'
    fold = 0
    model_string = 'chr'
    tid = 0
    save_path = f'{job_name}/{data_path}_{model_string}_{fold}/'
    obj = pickle.load(open(save_path + f'best_model_{tid}.p', "rb"))
    model = dill.loads(obj)
    data = np.load(data_path + f'/fold_{fold}.npy', allow_pickle=True).tolist()
    X_tr, Y_tr = data['tr']
    X_val, Y_val = data['val']
    X_test, Y_test = data['test']
    scaler = data['normalizer']
    column_names = data['column_names']


    chr = CHR(model, ymin=Y_val.min(), ymax=Y_val.max(), y_steps=200, delta_alpha=0.001, randomize=True)
    chr.calibrate(X_val, Y_val, alpha=0.1)
    bands = chr.predict(X_test)
    print(bands)
    plot_func(X_test[:, 0], Y_test, quantiles=bands, quantile_labels=["lower", "upper"],
              title="Test data with density-based prediction bands",filename='example')

