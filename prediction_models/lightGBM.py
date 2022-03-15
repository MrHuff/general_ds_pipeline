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
        validation_data =  lgb.Dataset(X_val, Y_val)
        self.model = lgb.train(params['lgbm_params'],
                               self.train_data,
                               num_boost_round=params['its'],
                               valid_sets=[validation_data],
                               early_stopping_rounds=100,
                               verbose_eval=True,
                               )

    def evaluate(self,X_val,Y_val):
        y_pred = self.model.predict(X_val,num_iteration=self.model.best_iteration)
        r2 = calc_r2(y_pred,Y_val)
        return r2

