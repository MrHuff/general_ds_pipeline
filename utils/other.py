import numpy as np
def calc_r2(y_pred,y_true):
    return 1-((y_true-y_pred)**2).mean()/y_true.var()
