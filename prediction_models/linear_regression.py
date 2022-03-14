import statsmodels.api as sm
from utils.other import calc_r2
class OLS_fitter():
    def __init__(self,X,Y):
        self.X = sm.add_constant(X)
        self.Y=Y

    def fit(self,hparams):
        self.model = sm.OLS(self.Y, self.X)
        results = self.model.fit()

    def evaluate(self,X_val,Y_val):
        y_pred = self.model.predict(X_val)
        r2 = calc_r2(y_pred,Y_val)
        return r2





