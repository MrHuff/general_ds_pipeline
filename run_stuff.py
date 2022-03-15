from utils.train_object import *
#run stuff make sure it does something. Then do the post processing and after that it should be gg
import pykeops
# pykeops.clean_pykeops()
if __name__ == '__main__':
    job_params={
        'data_path':'housing_no_normalization',
        'job_name':'test_2',
        'fold':0,
        'model_string':'ols',
        'hyperits':1,
        'epochs':100,
        'device':'cuda:0',
        'patience':25

    }
    # for m in ['ols','lasso','ridge','elastic','knn','GP','lightgbm','NN','chr']:
    for m in ['chr']:
        job_params['model_string'] = m
        c=train_object_regression(job_params=job_params)
        c.hypertune()
