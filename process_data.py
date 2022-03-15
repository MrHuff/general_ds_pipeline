import pandas as pd

from preprocessing_tools.load_normalize import *

if __name__ == '__main__':
    folder = 'housing'
    raw_df = pd.read_csv('archive/housing.csv',index_col=0)

    raw_df = raw_df.dropna()

    raw_df=pd.get_dummies(raw_df,columns=['ocean_proximity'])
    x_cols = raw_df.columns.to_list()
    x_cols.remove('median_house_value')
    X=raw_df[x_cols].values
    y=raw_df['median_house_value'].values

    split_normalize_save(folder,X,y,5,normalize_y=False)




