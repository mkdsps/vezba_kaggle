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

def export_longest_models(df, filename="top_50_models.txt"):
    # 1. Uzimamo samo jedinstvene nazive modela
    unique_models = df['model'].dropna().unique()
    
    # 2. Sortiramo po broju razmaka (count(" ")) u opadajućem redosledu
    # lambda x: x.count(" ") vraća broj razmaka u stringu
    sorted_models = sorted(unique_models, key=lambda x: str(x).count(" "), reverse=True)
    
    # 3. Uzimamo prvih 50
    top_50 = sorted_models[:200]
    
    # 4. Upisujemo u .txt fajl (svaki model u novi red)
    with open(filename, "w", encoding="utf-8") as f:
        for model in top_50:
            f.write(f"{model}\n")
            
    print(f"Uspešno izvezeno 50 modela u fajl: {filename}")
    return top_50

# Korišćenje:
top_list = export_longest_models(df_train)


# %%

import pandas as pd
import numpy as np
import re

def car_data_scientist_processor(df):
    # Radimo na kopiji da ne menjamo originalni df direktno dok ne budemo sigurni
    m = df['model'].fillna("").astype(str).str.upper()

    # 1. Čišćenje duplih brendova (npr. "AMG S 63 AMG S 63" -> "AMG S 63")
    def clean_doubles(text):
        words = text.split()
        return " ".join(dict.fromkeys(words))
    df['model_clean'] = m.apply(clean_doubles)

    # 2. Snaga i Performanse (HP i Turbo)
    df['hp_ref'] = m.str.extract(r'[PT](\d{2,3})').astype(float)
    # Dodatak za specifične snažne modele koje smo videli
    df['is_high_performance'] = m.str.contains('SVR|ST|RED|BLACKWING|COMPETITION|S-MODEL|TYPE S').astype(int)
    df['is_turbo_supercharged'] = m.str.contains('TURBO|SUPERCHARGED|2\.0T|3\.0T|4\.0T|2\.4T|4\.2L').astype(int)

    # 3. Električna snaga i domet
    df['is_electric_long_range'] = m.str.contains('LONG RANGE|P100D|RECHARGE|PURE ELECTRIC').astype(int)
    df['is_plugin_hybrid'] = m.str.contains('PLUG-IN|PHEV|HYBRID').astype(int)

    # 4. Specijalne i limitirane serije (VREDE VIŠE!)
    special_keywords = 'EDITION|HERITAGE|SPECIAL|ANNIVERSARY|FINAL|CRAFTED|PMC|FIRST EDITION'
    df['is_special_series'] = m.str.contains(special_keywords).astype(int)

    # 5. Truck/SUV specifiteti (Šasija i Vuča)
    df['is_heavy_duty'] = m.str.contains('HD|DRW|3500|2500|SUPER DUTY|H/D').astype(int)
    df['cab_type'] = m.str.extract(r'(CREW CAB|QUAD CAB|MEGA CAB|EXTENDED CAB|SUPERCAB)')
    df['cab_type'] = df['cab_type'].fillna('Standard').astype('category')

    # 6. Luksuzni paketi (High-End trimovi)
    luxury_keywords = 'AUTOBIOGRAPHY|WESTMINSTER|INSCRIPTION|LARIAT|ULTIMATE|PLATINUM|HIGH COUNTRY|DENALI'
    df['is_luxury_trim'] = m.str.contains(luxury_keywords).astype(int)

    # 7. Međuosovinsko rastojanje (L, SWB, LWB, ESV, XL)
    df['is_extended_length'] = m.str.contains(r'\bLWB\b|\bL\b|ESV|\bXL\b|EXTENDED').astype(int)

    return df

# df_train = car_data_scientist_processor(df_train)

# %%

print(df_all.columns)



# %%
