import matplotlib.pyplot as plt
import pickle
import dill
import numpy as np
import pandas as pd
import seaborn as sns
sns.set()
if __name__ == '__main__':
    data_path = 'housing_no_normalization'
    job_name = 'test_2'
    model_string = 'elastic'#['ols','lasso','ridge','elastic']
    tids= [0,0,0]
    coeffs_list =[]
    data = np.load(data_path + f'/fold_{0}.npy', allow_pickle=True).tolist()
    column_names = data['column_names']
    for fold,tid in zip([0,1,2],tids):
        save_path = f'{job_name}/{data_path}_{model_string}_{fold}/'
        obj = pickle.load(open(save_path + f'best_model_{tid}.p', "rb"))
        model = dill.loads(obj)
        if model_string=='ols':
            print(model.summary())
            coeffs=model.params[1:]
        else:
            coeffs=model.coef_
        coeffs_list.append(coeffs)
    data=pd.DataFrame(np.array(coeffs_list),columns=column_names)
    test=data.melt(value_vars=column_names)
    sns.set(rc={'figure.figsize': (15, 10)})

    ax = sns.boxplot(x="variable", y="value", data=test)
    plt.xticks(rotation=45)
    plt.show()
    # data = np.load(data_path + f'/fold_{fold}.npy', allow_pickle=True).tolist()
    # X_tr, Y_tr = data['tr']
    # X_val, Y_val = data['val']
    # X_test, Y_test = data['test']
    # scaler = data['normalizer']
    # column_names = data['column_names']