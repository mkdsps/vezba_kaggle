import pandas as pd
from features import add_features
from cleaning import clean_all

# ucitaj data

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

# spoji data u jedan df


df_train["_dataset"] = "train"
df_test["_dataset"] = "test"

df_all = pd.concat([df_train, df_test], axis=0, ignore_index=True)


# clean data

clean_df = clean_all(df_all)
# add_features

final_df = add_features(clean_df)

print(final_df.columns)

# priprema za trening

# 1. Razdvajanje na osnovu markera koji si napravio
df_train = final_df[final_df["_dataset"] == "train"].copy()
df_test = final_df[final_df["_dataset"] == "test"].copy()

# 2. Brisanje pomoćne kolone da ne smeta modelu
df_train = df_train.drop(columns=["_dataset", "id"])
df_test = df_test.drop(columns=["_dataset"])


# model_itd....

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np
from catboost import CatBoostRegressor, Pool

X = df_train.drop(columns=["price"])

print(X.isna().sum())

y = df_train["price"]

cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
print(cat_features)


######

from sklearn.model_selection import train_test_split

print("\n--- Započinjem finalni trening sa Early Stopping-om ---")

# 1. Izdvajamo 5% podataka samo da bi Early Stopping imao šta da gleda
X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
    X, 
    y, 
    test_size=0.05, 
    random_state=42
)

# 2. Inicijalizacija
final_model = CatBoostRegressor(
    iterations=5000,           # Možeš staviti više jer će ga stopirati
    learning_rate=0.02,
    depth=8,
    l2_leaf_reg=7,
    loss_function='RMSE',
    random_seed=42,
    task_type='GPU',
    devices='0',
    border_count=254,
    verbose=200
)

# 3. Fit sa validacionim setom za Early Stopping
final_model.fit(
    X_train_final, 
    y_train_final, 
    eval_set=(X_val_final, y_val_final),
    cat_features=cat_features,
    early_stopping_rounds=150  # Ovde dodaješ Early Stopping
)

X_test_final = df_test.drop(columns=['id'], errors="ignore")
X_test_final = X_test_final[X.columns]

test_preds = final_model.predict(X_test_final)

submission = pd.DataFrame({
    "id": df_test["id"],
    "price": test_preds
})

submission.to_csv("submission.csv", index=False)

print("Submission fajl 'submission.csv' je spreman za upload!")
