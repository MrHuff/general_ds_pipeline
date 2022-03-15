import torch
import numpy as np
from utils.other import *
from chr.custom_black_box import *
import tqdm

class chr_nn():
    def __init__(self,X_tr,Y_tr,dataloader=None):
        self.grid_quantiles = np.arange(0.01, 1.0, 0.01)
        self.X_tr = X_tr
        self.Y_tr = Y_tr
        self.dataloader=dataloader

    def fit(self, params):
        self.device = params['device']
        self.model = NNet(quantiles=self.grid_quantiles,num_features=params['x_in'],act_func=params['transformation'],
             n_layers=params['depth_x'],num_hidden=params['width_x'],dropout=params['dropout'],no_crossing=True)
        self.bbox =QNet(quantile_net=self.model,learning_rate=params['lr'],num_epochs=params['epochs'],batch_size=params['bs'],calibrate=1)
        self.bbox.fit(X=self.X_tr,Y=self.Y_tr)

    def validation_loop(self,mode='val'):
        self.model.eval()
        self.dataloader.dataset.set(mode)
        all_y_pred = []
        all_y = []
        for i,(X,y) in enumerate(tqdm.tqdm(self.dataloader)):
            X = X.to(self.device)
            y = y.to(self.device)
            with torch.no_grad():
                y_pred = self.model(X)
                y_pred = y_pred[:,50]
                all_y_pred.append(y_pred.cpu().numpy())
                all_y.append(y.cpu().numpy())
        all_y_pred=np.concatenate(all_y_pred,axis=0)
        all_y=np.concatenate(all_y,axis=0)
        return calc_r2(all_y_pred,all_y)

    def evaluate(self,mode):
        return self.validation_loop(mode)











