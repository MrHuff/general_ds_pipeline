from hyperopt import Trials
import dill
import pickle
import pandas as pd
if __name__ == '__main__':
    data_path = 'housing'
    job_name='test'
    h_its_method = [1, 5, 5, 5, 5, 2, 5, 5, 5]
    data_list=[]
    for model_string, hits in zip(['ols', 'lasso', 'ridge', 'elastic', 'knn', 'GP', 'lightgbm', 'NN', 'chr'], h_its_method):
        for fold in [0, 1, 2]:
            try:
                save_path = f'{job_name}/{data_path}_{model_string}_{fold}/'
                trials = pickle.load(open(save_path+'hyperopt_database.p', "rb"))
                trials = dill.loads(trials)
                best_trial = sorted(trials.trials, key=lambda x: x['result']['test_loss'], reverse=True)[0]
                best_tid = best_trial['tid']
                data_list.append([ model_string ,fold, best_tid,best_trial['result']['loss'], best_trial['result']['test_loss']])
            except Exception as e:
                print(e)
    df = pd.DataFrame(data_list,columns=['model','fold','best_it','val_loss','test_loss'])
    df.to_csv(f'{job_name}_complete_results.csv')
    summary_df = df.groupby(['model'])['test_loss'].mean()
    summary_df_std = df.groupby(['model'])['test_loss'].std()
    new_latex_df=summary_df.apply(lambda x: rf'${round(x,3)} \pm')+summary_df_std.apply(lambda x: f'{round(x,3)}$')
    new_latex_df.to_csv(f'{job_name}_final_results.csv')






