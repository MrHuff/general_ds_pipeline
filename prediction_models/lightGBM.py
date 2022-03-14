import lightgbm as lgb
import numpy as np

from utils.other import calc_r2
class lgbm_regression():
    def __init__(self,X_tr,Y_tr):
        self.X_tr = X_tr
        self.Y_tr = Y_tr
        self.train_data = lgb.Dataset(self.X_tr, label=self.Y_tr)

    def fit(self,params,val_data):
        X_val,Y_val = val_data
        validation_data =  lgb.Dataset(X_val, label=Y_val)
        bst = lgb.train(params['hparams'], self.train_data, params['num_round'], valid_sets=[validation_data])
        # pass
        # self.model = ElasticNet(alpha=hparams['alpha'])
        # self.model.fit(self.X_tr, self.Y_tr)


    def evaluate(self,X_val,Y_val):
        pass
        y_pred = self.model.predict(X_val)
        r2 = calc_r2(y_pred,Y_val)
        return r2

