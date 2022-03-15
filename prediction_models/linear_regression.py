import statsmodels.api as sm
from utils.other import calc_r2
class OLS_fitter():
    def __init__(self,X,Y):
        self.X = sm.add_constant(X,has_constant='add')
        self.Y=Y

    def fit(self,hparams):
        tmp = sm.OLS(self.Y, self.X)
        self.model = tmp.fit()
        print(self.model.summary())

    def evaluate(self,X_val,Y_val):
        X_in = sm.add_constant(X_val,has_constant='add')
        y_pred = self.model.predict(X_in)
        r2 = calc_r2(y_pred,Y_val)
        return r2





