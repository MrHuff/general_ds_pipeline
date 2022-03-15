import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn_pandas import DataFrameMapper
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import os

class StratifiedKFold3(StratifiedKFold):
    def split(self, X, y, groups=None,stratify=False):
        s = super().split(X, y, groups)
        fold_indices=[]
        for train_indxs, test_indxs in s:
            y_train = y[train_indxs]
            if stratify:
                train_indxs, cv_indxs = train_test_split(train_indxs,stratify=y_train, test_size=(1 / (self.n_splits - 1)))
            else:
                train_indxs, cv_indxs = train_test_split(train_indxs, test_size=(1 / (self.n_splits - 1)))

            # yield train_indxs, cv_indxs, test_indxs
            fold_indices.append((train_indxs, cv_indxs, test_indxs))
        return fold_indices


def split_normalize_save(savedirname,X,y,folds,normalize_y = False):
    if not os.path.exists(savedirname):
        os.makedirs(savedirname)
    indices = np.arange(y.shape[0])
    X,y = X.astype(float),y.astype(float)
    for f in range(folds):
        scaler = StandardScaler()
        scaler_y = StandardScaler()
        tr_ind, val_ind, test_ind = StratifiedKFold3(folds).split(indices, y)[f]
        X_tr = X[tr_ind,:]
        X_val = X[val_ind,:]
        X_test = X[test_ind,:]
        y_tr = y[tr_ind]
        y_val = y[val_ind]
        y_test = y[test_ind]
        if normalize_y:
            y_tr = scaler_y.fit_transform(y_tr[:,np.newaxis])
            y_val = scaler_y.transform(y_val[:,np.newaxis])
            y_test = scaler_y.transform(y_test[:,np.newaxis])
            y_tr = y_tr.squeeze()
            y_val = y_val.squeeze()
            y_test = y_test.squeeze()

        X_tr = scaler.fit_transform(X_tr)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        with open(f'{savedirname}/fold_{f}.npy', 'wb') as f:
            np.save(f, {'tr':[X_tr,y_tr],'val':[X_val,y_val],'test':[X_test,y_test], 'normalizer': scaler_y})










