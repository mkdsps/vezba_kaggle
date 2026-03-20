# %%

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

# %%

counts = df_all['brand'].value_counts()

top5_rarest = counts.nsmallest(20)

print(df_all['brand'].nunique())

print(top5_rarest)



# %%

import numpy as np

def add_brand_features_safe(df_train, df_test):

    brand_stats = df_train.groupby('brand').agg(
        train_brand_count=('price', 'count'),
        train_brand_median=('price', 'median')
    ).reset_index()

    def apply_stats(target_df):
        res = target_df.merge(brand_stats, on='brand', how='left')
        
        # svakako nemamo ali neka ostane....
        res['train_brand_count'] = res['train_brand_count'].fillna(1)
        res['train_brand_median'] = res['train_brand_median'].fillna(df_train['price'].median())

        res['brand_count_log'] = np.log1p(res['train_brand_count'])
        
        ratio = res['train_brand_median'] / res['train_brand_count']
        res['brand_exclusivity_log'] = np.log1p(ratio)

 
        return res.drop(columns=['train_brand_count', 'train_brand_median'])

    # Primena
    df_train_new = apply_stats(df_train)
    df_test_new = apply_stats(df_test)
    
    return df_train_new, df_test_new

df_train_new, df_test_new = add_brand_features_safe(df_train, df_test)

# %%


# %%

