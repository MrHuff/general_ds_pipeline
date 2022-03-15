
import matplotlib.pyplot as plt
import pickle
import dill
import numpy as np
import gpytorch
import torch

#NORMALIZE Y TO GET SENSIBLE BOUNDS!

if __name__ == '__main__':
    data_path = 'housing'
    job_name = 'test'
    fold = 0
    model_string = 'GP'
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
    model.eval()
    likelihood = model.likelihood
    likelihood.eval()
    device ='cuda:0'
    X_in = torch.from_numpy(X_test).float().to(device)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(X_in))
        mean = observed_pred.mean
        lower, upper = observed_pred.confidence_region()

    mean = mean.cpu()
    lower = lower.cpu()
    upper = upper.cpu()

    with torch.no_grad():

        f, ax = plt.subplots(1, 1, figsize=(12, 9))
        col=0
        n_to_plot=10
        plot_covar = X_test[:,col]
        sort_ind = np.argsort(plot_covar)
        plot_covar=plot_covar[sort_ind]
        plot_covar=plot_covar[:n_to_plot]
        # Plot training data as black stars
        # ax.plot(X_test[col].numpy(), train_y.numpy(), 'k*')
        # Plot predictive means as blue line
        ax.plot(plot_covar, mean.numpy()[sort_ind][:n_to_plot], 'b')
        # Shade between the lower and upper confidence bounds
        # print(lower.numpy()[sort_ind][:n_to_plot], upper.numpy()[sort_ind][:n_to_plot])
        ax.fill_between(plot_covar, lower.numpy()[sort_ind][:n_to_plot], upper.numpy()[sort_ind][:n_to_plot], alpha=0.5)
        # ax.set_ylim([-3, 3])
        ax.legend(['Observed Data', 'Mean', 'Confidence'])

        plt.show()


