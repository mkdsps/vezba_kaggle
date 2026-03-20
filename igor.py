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

print(df_train['model'].nunique())
print(df_train['model'].isna().sum())

# %%


import pandas as pd
import numpy as np
import re

def car_data_scientist_processor(df):
    df = df.copy()
    
    # Prebacujemo u string i upper za lakšu pretragu
    m = df['model'].fillna("MISSING").astype(str).str.upper()
    e = df['engine'].fillna("MISSING").astype(str).str.upper()

    def clean_doubles(text):
        words = text.split()
        return " ".join(dict.fromkeys(words))
    df['model_clean'] = m.apply(clean_doubles)

    perf_pattern = 'SVR|ST|RED|BLACKWING|COMPETITION|S-MODEL|TYPE S|AMG|M-SPORT|RS'
    df['is_high_performance'] = m.str.contains(perf_pattern).astype(int)
    
    turbo_pattern = 'TURBO|SUPERCHARGED|2\.0T|3\.0T|4\.0T|2\.4T|4\.2L'
    m_turbo = m.str.contains(turbo_pattern).astype(int)
    e_turbo = e.str.contains('TURBO|SUPER').astype(int)
    df['is_forced_induction'] = ((m_turbo == 1) | (e_turbo == 1)).astype(int)

    df['is_electric_long_range'] = m.str.contains('LONG RANGE|P100D|RECHARGE|PURE ELECTRIC').astype(int)
    df['is_plugin_hybrid'] = (m.str.contains('PLUG-IN|PHEV|HYBRID') | e.str.contains('HYBRID')).astype(int)

    special_keywords = 'EDITION|HERITAGE|SPECIAL|ANNIVERSARY|FINAL|CRAFTED|PMC|FIRST EDITION'
    df['is_special_series'] = m.str.contains(special_keywords).astype(int)

    luxury_keywords = 'AUTOBIOGRAPHY|WESTMINSTER|INSCRIPTION|LARIAT|ULTIMATE|PLATINUM|HIGH COUNTRY|DENALI'
    df['is_luxury_trim'] = m.str.contains(luxury_keywords).astype(int)
 
    df['is_extended_length'] = m.str.contains(r'\bLWB\b|\bL\b|ESV|\bXL\b|EXTENDED').astype(int)


    positive_cols = [
        'is_high_performance', 'is_forced_induction', 'is_electric_long_range',
        'is_plugin_hybrid', 'is_special_series', 
        'is_luxury_trim', 'is_extended_length' ]    
    
    df['positive_features_count'] = df[positive_cols].sum(axis=1)

    return df

df_train = car_data_scientist_processor(df_train)

# %%
import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_importance_check(df):
    # Postavljamo stil
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(15, 6))

    # 1. Grafik: Prosečna cena po Luxury Score-u
    plt.subplot(1, 2, 1)
    # Koristimo originalnu cenu za lakše razumevanje (ne log)
    sns.barplot(x='positive_features_count', y='price', data=df, palette='viridis')
    plt.title('Prosečna cena u odnosu na Luxury Score', fontsize=14)
    plt.xlabel('Broj pozitivnih feature-a (Luxury Score)', fontsize=12)
    plt.ylabel('Prosečna Cena', fontsize=12)

    # 2. Grafik: Korelacija novih feature-a sa cenom
    plt.subplot(1, 2, 2)
    new_features = ['is_forced_induction', 'is_high_performance', 'is_luxury_trim',  
                     'positive_features_count', 'positive_features_count', 'price']
    
    # Računamo korelaciju samo za ove kolone
    corr = df[new_features].corr()
    sns.heatmap(corr[['price']].sort_values(by='price', ascending=False), 
                annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Korelacija novih feature-a sa cenom', fontsize=14)

    plt.tight_layout()
    plt.savefig("slike/model_features.png")
    plt.show()

# Pokreni analizu na train setu
plot_feature_importance_check(df_train)

# %%

def process_year_features(df):
    df = df.copy()
    
    current_year = 2024
    
    df['car_age'] = (current_year - df['model_year']).clip(lower=0)
    
    df['car_age_squared'] = df['car_age'] ** 2
    
    df['log_car_age'] = np.log1p(df['car_age'])
    
    if 'positive_features_count' in df.columns:
        df['luxury_score_per_age'] = df['positive_features_count'] / (df['car_age'] + 1)
        
    def get_car_era(age):
        if age <= 2: return 'NEW'           # Miris novog auta
        if age <= 5: return 'MODERN'        # Još uvek aktuelan model
        if age <= 12: return 'USED'         # Standardni polovnjak
        if age <= 25: return 'OLD'          # Na ivici rashoda
        return 'VINTAGE'                    # Potencijalni klasik
        
    df['car_era'] = df['car_age'].apply(get_car_era)
    
    return df


df_train = process_year_features(df_train)

# %%

print(df_train.columns)


# %%

def process_mileage_features(df):
    df = df.copy()
    
    df['log_mileage'] = np.log1p(df['milage'])
    
    df['mileage_per_year'] = df['milage'] / (df['car_age'] + 1)
    
    df['mileage_squared'] = df['milage'] ** 2
    
    def mileage_category(m):
        if m < 15000: return 'NEW_LIKE'
        if m < 60000: return 'LOW_MILEAGE'
        if m < 150000: return 'AVERAGE'
        if m < 250000: return 'HIGH_MILEAGE'
        return 'VERY_HIGH_MILEAGE'
    
    df['mileage_type'] = df['milage'].apply(mileage_category)
    
    df['is_high_usage'] = (df['mileage_per_year'] > 20000).astype(int)
    
    return df

df_train = process_mileage_features(df_train)

# %%


import matplotlib.pyplot as plt
import seaborn as sns

def print_mileage_correlations(df):
    # Lista svih mileage feature-a koje smo napravili
    mileage_cols = [
        'milage',                # Originalna kolona (proveri da li je milage ili mileage)
        'log_mileage',           # Logaritamska (linearnost)
        'mileage_per_year',      # Intenzitet vožnje
        'mileage_squared',       # Nelinearni pad
        'is_high_usage',         # Binarni (preko 20k/god)
        'price'                  # Target
    ]
    
    # Filtriramo samo one koji stvarno postoje u df da ne pukne kod
    existing_cols = [c for c in mileage_cols if c in df.columns]
    
    # Računamo korelaciju
    corrs = df[existing_cols].corr()['price'].sort_values(ascending=False)
    
    print("-" * 30)
    print("KORELACIJA SA CENOM (Price):")
    print("-" * 30)
    print(corrs)
    print("-" * 30)

# Pokreni na train setu
print_mileage_correlations(df_train)


# %%
