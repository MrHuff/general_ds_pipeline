
import gpytorch
import torch
import tqdm
from utils.other import *

class ExactGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.keops.RBFKernel(ard_num_dims=train_x.shape[1]))

    def forward(self,X):
        mean_x = self.mean_module(X)
        covar_x = self.kernel(X)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)



class GP_regression():
    def __init__(self,X_tr,Y_tr):
        self.X_tr = torch.from_numpy(X_tr).float()
        self.Y_tr = torch.from_numpy(Y_tr).float()
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = ExactGP(likelihood=self.likelihood, train_x=self.X_tr, train_y=self.Y_tr)

    def fit(self,hparams):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=hparams['lr'])  # Includes GaussianLikelihood parameters
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        pbar = tqdm.tqdm(range(hparams['epochs']))
        self.device = hparams['device']
        self.X_tr = self.X_tr.to(self.device)
        self.Y_tr = self.Y_tr.to(self.device)
        self.model = self.model.to(self.device)
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
        return loss.item(), self.model.kernel.base_kernel.lengthscale, self.model.likelihood.noise.item()

        # self.model = KNeighborsRegressor(n_neighbors=hparams['k'])
        # self.model.fit(self.X_tr, self.Y_tr)

    def evaluate(self,X_val,Y_val):
        self.model.eval()
        self.likelihood.eval()
        X_in = torch.from_numpy(X_val).float().to(self.device)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(X_in))
            mean = observed_pred.mean
            lower, upper = observed_pred.confidence_region()
        mean = mean.cpu().numpy()
        r2 = calc_r2(mean,Y_val)
        return r2