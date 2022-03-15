import torch
import dill
import pickle
import os
from hyperopt import hp,tpe,Trials,fmin,space_eval,STATUS_OK,STATUS_FAIL,rand
import numpy as np
from prediction_models.CHR import *
from prediction_models.GP import *
from prediction_models.kNN import *
from prediction_models.lightGBM import *
from prediction_models.linear_regression import *
from prediction_models.reg_regression import *
from prediction_models.NN import *


class general_chunk_iterator():
    def __init__(self,X,y,shuffle,batch_size):
        self.X = X
        self.y = y
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.n = self.X.shape[0]
        self.chunks=self.n//batch_size+1
        self.perm = torch.randperm(self.n)
        if self.shuffle:
            self.X = self.X[self.perm,:]
            self.y = self.y[self.perm,:]
        self._index = 0
        self.it_X = torch.chunk(self.X,self.chunks)
        self.it_y = torch.chunk(self.y,self.chunks)
        self.true_chunks = len(self.it_X)

    def __next__(self):
        ''''Returns the next value from team object's lists '''
        if self._index < self.true_chunks:
            result = (self.it_X[self._index],self.it_y[self._index])
            self._index += 1
            return result
        # End of Iteration
        raise StopIteration

    def __len__(self):
        return len(self.it_X)

class custom_dataloader():
    def __init__(self,dataset,batch_size=32,shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n = self.dataset.X.shape[0]
        self.len=self.n//batch_size+1
    def __iter__(self):
        return general_chunk_iterator(X =self.dataset.X,
                              y = self.dataset.y,
                              shuffle = self.shuffle,
                              batch_size=self.batch_size,
                              )
    def __len__(self):
        self.n = self.dataset.X.shape[0]
        self.len = self.n // self.batch_size + 1
        return self.len

class general_dataset():
    def __init__(self,X_tr,y_tr,X_val,y_val,X_test,y_test):
        self.train_X=torch.from_numpy(X_tr).float()
        self.train_y=torch.from_numpy(y_tr).float()
        self.val_X=torch.from_numpy(X_val).float()
        self.val_y=torch.from_numpy(y_val).float()
        self.test_X=torch.from_numpy(X_test).float()
        self.test_y=torch.from_numpy(y_test).float()

    def set(self, mode='train'):
        self.X = getattr(self, f'{mode}_X')
        self.y = getattr(self, f'{mode}_y')

    def __getitem__(self, index):
        return self.X[index, :], self.y[index]

    def __len__(self):
        return self.X.shape[0]


class train_object_regression():
    def __init__(self,job_params):
        self.global_hyperit=0
        self.data_path=job_params['data_path']
        self.job_name=job_params['job_name']
        self.fold=job_params['fold']
        self.model_string = job_params['model_string']
        self.save_path = f'{self.job_name}/{self.data_path}_{self.model_string}_{self.fold}/'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        data=np.load(self.data_path+f'/fold_{self.fold}.npy',allow_pickle=True).tolist()
        self.X_tr,self.Y_tr = data['tr']
        self.X_val,self.Y_val = data['val']
        self.X_test,self.Y_test = data['test']


        self.dataset=general_dataset(self.X_tr,self.Y_tr,
                                     self.X_val, self.Y_val,
                                     self.X_test, self.Y_test
                                     )
        self.dataset.set('train')
        self.dataloader=custom_dataloader(self.dataset)
        self.job_params=job_params

        if self.model_string == 'ols':
            self.hyperopt_params =['placeholder']
            hparams={'placeholder':[0,1]}
        if self.model_string == 'lasso':
            self.hyperopt_params = ['alpha']
            hparams = {'alpha': np.linspace(0,1,100).tolist()}
        if self.model_string == 'ridge':
            self.hyperopt_params = ['alpha']
            hparams = {'alpha': np.linspace(0, 1, 100).tolist()}
        if self.model_string == 'elastic':
            self.hyperopt_params = ['alpha']
            hparams = {'alpha': np.linspace(0, 1, 100).tolist()}
        if self.model_string == 'knn':
            self.hyperopt_params = ['k']
            hparams = {'k': list(range(1,50))}
        if self.model_string == 'GP':
            self.hyperopt_params = ['lr']
            hparams = {'lr': [1e-3, 1e-2,1e-1,1.0]}

        if self.model_string == 'NN':
            self.hyperopt_params = ['transformation', 'depth_x', 'width_x', 'bs', 'lr']
            hparams = {
                'depth_x': [1,2,3,4],
                'width_x': [32,64,128],
                'bs': [100,200,250,500,1000],
                'lr': [1e-2,1e-3,1e-1],
                'transformation': [torch.tanh,torch.relu],
            }

        if self.model_string == 'chr':
            self.hyperopt_params = ['transformation', 'depth_x', 'width_x', 'bs', 'lr','dropout']
            hparams = {
                'depth_x': [1, 2, 3, 4],
                'width_x': [32, 64, 128],
                'bs': [100, 200, 250, 500,1000],
                'lr': [1e-2,1e-3,1e-1],
                'transformation': [torch.nn.Tanh(), torch.nn.ReLU()],
                'dropout':[0.0,0.1,0.2,0.3]
            }

        if self.model_string == 'lightgbm':
            self.hyperparameter_space = {
                'num_leaves': hp.quniform('num_leaves', 7, 4095, 1),
                'min_data_in_leaf': hp.quniform('min_data_in_leaf', 10, 30, 1),
                'learning_rate': hp.loguniform('learning_rate', -5.0, -2.3),
                'min_sum_hessian_in_leaf': hp.loguniform('min_sum_hessian_in_leaf', 0, 2.3),
                'bagging_freq': hp.quniform('bagging_freq', 1, 5, 1),
                'lambda_l1': hp.uniform('lambda_l1', 0, 10),
                'lambda_l2': hp.uniform('lambda_l2', 0, 10),
            }
            self.hyperparameter_space['feature_fraction'] = hp.uniform('feature_fraction', 0.75, 1.0)
            self.hyperparameter_space['bagging_fraction'] = hp.uniform('bagging_fraction', 0.75, 1.0)
            self.hyperparameter_space['max_bin'] = hp.quniform('max_bin', 12, 256, 1)

        if self.model_string!='lightgbm':
            self.get_hyperparameterspace(hparams)

    def get_hyperparameterspace(self,hyper_param_space):
        self.hyperparameter_space = {}
        for string in self.hyperopt_params:
            self.hyperparameter_space[string] = hp.choice(string, hyper_param_space[string])

    def __call__(self,params):

        if self.model_string == 'ols':
            m = OLS_fitter(X=self.X_tr,Y=self.Y_tr)
            m.fit(params)

        if self.model_string == 'lasso':
            m = lasso_regression(X_tr=self.X_tr,Y_tr=self.Y_tr)
            m.fit(params)

        if self.model_string == 'ridge':
            m = ridge_regression(X_tr=self.X_tr,Y_tr=self.Y_tr)
            m.fit(params)

        if self.model_string == 'elastic':
            m = elastic_regression(X_tr=self.X_tr,Y_tr=self.Y_tr)
            m.fit(params)

        if self.model_string == 'knn':
            m = knn_regression(X_tr=self.X_tr,Y_tr=self.Y_tr)
            m.fit(params)

        if self.model_string == 'GP':
            params['epochs'] = self.job_params['epochs']
            params['device'] = self.job_params['device']


            m = GP_regression(X_tr=self.X_tr,Y_tr=self.Y_tr)
            m.fit(params)

        if self.model_string == 'NN':
            m = nn_regression(self.dataloader)
            params['epochs'] = self.job_params['epochs']
            params['device'] = self.job_params['device']
            nn_params = {
                'd_in_x': self.X_tr.shape[1],
                'cat_size_list': [],
                'output_dim': 1,
                'transformation': params['transformation'],
                'layers_x': [params['width_x']] * params['depth_x'],
            }
            params['nn_params'] = nn_params
            params['patience'] = self.job_params['patience']

            m.fit(params)

        if self.model_string == 'chr':
            m = chr_nn(X_tr=self.X_tr,Y_tr=self.Y_tr,dataloader=self.dataloader)
            params['epochs'] = self.job_params['epochs']
            params['device'] = self.job_params['device']
            params['patience'] = self.job_params['patience']
            params['x_in'] = self.X_tr.shape[1]
            m.fit(params)

        if self.model_string == 'lightgbm':
            lgbm_params=self.get_lgm_params(params)

            params['lgbm_params']=lgbm_params
            params['its']=1000

            m = lgbm_regression(X_tr=self.X_tr,Y_tr=self.Y_tr)
            m.fit(params,(self.X_val,self.Y_val))

        if self.model_string in ['ols','lasso','ridge','elastic','knn','GP','lightgbm']:
            val_error = m.evaluate(self.X_val,self.Y_val)
            test_error = m.evaluate(self.X_test,self.Y_test)
        else:
            val_error = m.evaluate('val')
            test_error = m.evaluate('test')
        model_copy = dill.dumps(m.model)

        pickle.dump(model_copy,
                    open(self.save_path+f'best_model_{self.global_hyperit}.p',
                         "wb"))

        self.global_hyperit+=1
        return  {'loss': val_error,
                'status': STATUS_OK,
                'test_loss': test_error,
                 'net_params':params
                }



    def get_lgm_params(self,space):
        lgb_params = dict()
        lgb_params['boosting_type'] = space['boosting_type'] if 'boosting_type' in space else 'gbdt'
        lgb_params['application'] = 'regression'
        lgb_params['metric'] = 'mse'
        lgb_params['num_class'] = 1
        lgb_params['learning_rate'] = space['learning_rate']
        lgb_params['num_leaves'] = int(space['num_leaves'])
        lgb_params['min_data_in_leaf'] = int(space['min_data_in_leaf'])
        lgb_params['min_sum_hessian_in_leaf'] = space['min_sum_hessian_in_leaf']
        lgb_params['max_depth'] = -1
        lgb_params['lambda_l1'] = space['lambda_l1'] if 'lambda_l1' in space else 0.0
        lgb_params['lambda_l2'] = space['lambda_l2'] if 'lambda_l2' in space else 0.0
        lgb_params['bagging_freq'] = int(space['bagging_freq']) if 'bagging_freq' in space else 1
        lgb_params['max_bin'] = int(space['max_bin']) if 'max_bin' in space else 256
        lgb_params['feature_fraction'] = space['feature_fraction']
        lgb_params['bagging_fraction'] = space['bagging_fraction']
        return lgb_params

    def hypertune(self):
        if os.path.exists(self.save_path + 'hyperopt_database.p'):
            return
        trials = Trials()
        best = fmin(fn=self,
                    space=self.hyperparameter_space,
                    algo=tpe.suggest,
                    max_evals=self.job_params['hyperits'],
                    trials=trials,
                    verbose=True)
        print(space_eval(self.hyperparameter_space, best))
        model_copy = dill.dumps(trials)
        pickle.dump(model_copy,
                    open(self.save_path + 'hyperopt_database.p',
                         "wb"))



# class train_object_classification():
#     def __init__(self,job_params,hparamspace):
#         pass
#         #load appropriate data
#         #get dataloaders if necessary
#         #define hparamspace given a model
#
#     def __call__(self):
#         pass
#         #call
#
#     def hypertune(self):
#         pass
#






