
import gpytorch
import torch
import tqdm
from utils.other import *
class ExactGPGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.keops.RBFKernel(ard_num_dims=train_x.shape[1]))

    def forward(self,X):
        mean_x = self.mean_module(X)
        covar_x = self.kernel(X)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GP_regression():
    def __init__(self,X_tr,Y_tr):
        self.X_tr = torch.from_numpy(X_tr)
        self.Y_tr = torch.from_numpy(Y_tr)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = ExactGPGP(likelihood=self.likelihood, train_x=self.X_tr, train_y=self.Y_tr)

    def fit(self,hparams):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        pbar = tqdm.tqdm(range(hparams['epochs']))
        self.X_tr = self.X_tr.to(hparams['device'])
        self.Y_tr = self.Y_tr.to(hparams['device'])
        for i, j in enumerate(pbar):
            l, ls, noise = self.full_gp_loop(optimizer, mll)
            pbar.set_description(f"loss: {l} ls: {ls} noise: {noise}")

    def full_gp_loop(self, optimizer, mll):
        self.model.train()
        self.likelihood.train()
        optimizer.zero_grad()
        # Output from model
        output = self.model(self.X_tr)
        # Calc loss and backprop gradients
        loss = -mll(output, self.Y_tr)
        loss.backward()
        optimizer.step()
        return loss.item(), self.model.covar_module.base_kernel.lengthscale.item(), self.model.likelihood.noise.item()

        # self.model = KNeighborsRegressor(n_neighbors=hparams['k'])
        # self.model.fit(self.X_tr, self.Y_tr)

    def evaluate(self,X_val,Y_val):
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(X_val))
            mean = observed_pred.mean
            lower, upper = observed_pred.confidence_region()
        mean = mean.cpu()
        r2 = calc_r2(mean,Y_val)