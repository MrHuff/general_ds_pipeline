import copy

import numpy as np
import torch
from utils.other import *
import tqdm

class multi_input_Sequential(torch.nn.Sequential):
    def forward(self, inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs

class multi_input_Sequential_res_net(torch.nn.Sequential):
    def forward(self, inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                output = module(inputs)
                if inputs.shape[1]==output.shape[1]:
                    inputs = inputs+output
                else:
                    inputs = output
        return inputs

class nn_node(torch.nn.Module): #Add dropout layers, Do embedding layer as well!
    def __init__(self,d_in,d_out,cat_size_list,transformation=torch.tanh):
        super(nn_node, self).__init__()

        self.has_cat = len(cat_size_list)>0
        self.latent_col_list = []
        print('cat_size_list',cat_size_list)
        for i,el in enumerate(cat_size_list):
            col_size = el//2+2
            setattr(self,f'embedding_{i}',torch.nn.Embedding(el,col_size))
            self.latent_col_list.append(col_size)
        self.w = torch.nn.Linear(d_in+sum(self.latent_col_list),d_out)
        self.f = transformation
        # self.dropout = torch.nn.Dropout(dropout)
        self.bn = torch.nn.BatchNorm1d(d_out)

    def forward(self,X,x_cat=[]):
        if not isinstance(x_cat,list):
            seq = torch.unbind(x_cat,1)
            cat_vals = [X]
            for i,f in enumerate(seq):
                o = getattr(self,f'embedding_{i}')(f)
                cat_vals.append(o)
            X = torch.cat(cat_vals,dim=1)
        return self.bn(self.f(self.w(X)))

class feature_map(torch.nn.Module):
    def __init__(self,
                 d_in_x,
                 cat_size_list,
                 layers_x,
                 transformation=torch.tanh,
                 output_dim=10,
                 ):
        super(feature_map, self).__init__()
        self.output_dim=output_dim
        self.init_covariate_net(d_in_x,layers_x,cat_size_list,transformation,output_dim)

    def identity_transform(self, x):
        return x

    def init_covariate_net(self,d_in_x,layers_x,cat_size_list,transformation,output_dim):
        module_list = [nn_node(d_in=d_in_x,d_out=layers_x[0],cat_size_list=cat_size_list,transformation=transformation)]
        for l_i in range(1,len(layers_x)):
            module_list.append(nn_node(d_in=layers_x[l_i-1],d_out=layers_x[l_i],cat_size_list=[],transformation=transformation))
        self.covariate_net = multi_input_Sequential_res_net(*module_list)
        self.final_layer = torch.nn.Linear(layers_x[-1],output_dim)

    def forward(self,x_cov,x_cat=[]):
        return self.final_layer(self.covariate_net((x_cov,x_cat)))


class nn_regression():
    def __init__(self,dataloader):
        self.dataloader=dataloader
        self.loss = torch.nn.MSELoss()

    def fit(self,params):
        self.device = params['device']
        self.model = feature_map(**params['nn_params']).to(self.device)
        opt=torch.optim.Adam(self.model.parameters(),lr=params['lr'])
        best=-np.inf
        count=0
        for i in range(params['epochs']):
            self.train_loop(opt)
            r2=self.validation_loop('val')
            if r2>best:
                self.best_model = copy.deepcopy(self.model)
                best = r2
                print(r2)
                count=0
            else:
                count+=1
            if count>params['patience']:
                self.model = self.best_model
                return
        self.model = self.best_model

    def train_loop(self,opt):
        self.dataloader.dataset.set('train')
        for i,(X,y) in enumerate(tqdm.tqdm(self.dataloader)):
            X=X.to(self.device)
            y=y.to(self.device)
            y_pred=self.model(X)
            tot_loss = self.loss(y_pred.squeeze(),y.squeeze())
            opt.zero_grad()
            tot_loss.backward()
            opt.step()

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
                all_y_pred.append(y_pred.cpu().numpy())
                all_y.append(y.cpu().numpy())
        all_y_pred=np.concatenate(all_y_pred,axis=0)
        all_y=np.concatenate(all_y,axis=0)
        return calc_r2(all_y_pred.squeeze(),all_y.squeeze())

    def evaluate(self,mode):
        return self.validation_loop(mode)
