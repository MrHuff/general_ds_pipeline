from utils.train_object import *
#run stuff make sure it does something. Then do the post processing and after that it should be gg
import pykeops
# pykeops.clean_pykeops()
#Ok for CHR, there might be some pathologies pertaining to data normalization when using the tanh(), since it essentially squeezes the data.


if __name__ == '__main__':
    job_params={
        'data_path':'housing',
        'job_name':'test',
        'fold':0,
        'model_string':'ols',
        'hyperits':1,
        'epochs':100,
        'device':'cuda:0',
        'patience':25
    }
    # h_its_method=[1,20,20,20,20,2,20,20,20]
    # h_its_method=[1,5,5,5,5,2,5,5,5]
    h_its_method=[1]*9
    # for f in [0,1,2]:
    for f in [0]:
        for m,hits in zip(['ols','lasso','ridge','elastic','knn','GP','lightgbm','NN','chr'],h_its_method):
            job_params['model_string'] = m
            job_params['hyperits'] = hits
            job_params['fold'] = f
            c=train_object_regression(job_params=job_params)
            c.hypertune()
