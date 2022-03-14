from sklearn.linear_model import Ridge,Lasso,ElasticNet
from utils.other import calc_r2

class ridge_regression():
    def __init__(self,X_tr,Y_tr):
        self.X_tr = X_tr
        self.Y_tr = Y_tr

    def fit(self,hparams):
        self.model = Ridge(alpha=hparams['alpha'])
        self.model.fit(self.X_tr, self.Y_tr)
    def evaluate(self,X_val,Y_val):
        y_pred = self.model.predict(X_val)
        r2 = calc_r2(y_pred,Y_val)
        return r2
class lasso_regression():
    def __init__(self,X_tr,Y_tr):
        self.X_tr = X_tr
        self.Y_tr = Y_tr

    def fit(self,hparams):
        self.model = Lasso(alpha=hparams['alpha'])
        self.model.fit(self.X_tr, self.Y_tr)
    def evaluate(self,X_val,Y_val):
        y_pred = self.model.predict(X_val)
        r2 = calc_r2(y_pred,Y_val)
        return r2

class elastic_regression():
    def __init__(self,X_tr,Y_tr):
        self.X_tr = X_tr
        self.Y_tr = Y_tr

    def fit(self,hparams):
        self.model = ElasticNet(alpha=hparams['alpha'])
        self.model.fit(self.X_tr, self.Y_tr)
    def evaluate(self,X_val,Y_val):
        y_pred = self.model.predict(X_val)
        r2 = calc_r2(y_pred,Y_val)
        return r2