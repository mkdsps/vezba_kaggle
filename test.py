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

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np
from catboost import CatBoostRegressor, Pool

X = df_train.drop(columns=['price'])
y = df_train['price']

cat_features = ['brand', 'model', 'brand_class', 'cab_type', 'is_4wd', 'is_luxury_trim'] 

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []

print("Započinjem 5-Fold Cross-Validation...")
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Inicijalizacija modela sa preporučenim parametrima
    model = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.05,
        depth=6,
        loss_function='RMSE',
        random_seed=42,
    )
    
    # Trening
    model.fit(
        X_tr, y_tr,
        eval_set=(X_val, y_val),
        cat_features=cat_features,
        early_stopping_rounds=100
    )
    
    # Predikcija i računanje greške
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    scores.append(rmse)
    
    print(f"Fold {fold+1} RMSE: {rmse:.4f}")

print(f"\nProsečan RMSE (Log skala): {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")



