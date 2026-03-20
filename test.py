import pandas as pd
from features import add_features
from cleaning import clean_all

# ucitaj data

df_train = pd.read_csv('train.csv')
df_test  = pd.read_csv('test.csv')

# spoji data u jedan df 


df_train['_dataset'] = 'train'
df_test['_dataset']  = 'test'

df_all = pd.concat([df_train, df_test], axis = 0, ignore_index=True)

print(df_all.head())

# clean data 

clean_df = clean_all(df_all)

# add_features

final_df = add_features(clean_df)

# model_itd....
