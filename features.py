import pandas as pd
import re
import numpy as np


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    vraca kompletan dataset sa dodatim features i izbrisanim ostalim stvarima ( spojeni train i test )
    """
    df = add_engine_features(df)
    df = add_transmission_features(df)
    df = add_brand_features_safe(df)
    df = process_model_features(df)
    df = process_year_features(df)
    df = process_mileage_features(df)
    df = create_color_features(df)

    cols_to_drop = [
        'is_cvt', 
        'is_high_usage', 
        'is_dct', 
        'Electrical', 
        'is_manual', 'is_gasoline', 'is_automatic', 'is_plugin', 'is_turbo','milage','milage_squared'
    ]

    return df.drop(columns=cols_to_drop, errors='ignore')
    return df


def add_transmission_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["transmission"] = df["transmission"].fillna("missing").astype(str).str.strip()

    t = df["transmission"].str.lower()

    df["is_automatic"] = t.str.contains(
        r"automatic|\ba/t\b", regex=True, na=False
    ).astype(int)
    df["is_manual"] = t.str.contains(r"manual|\bm/t\b", regex=True, na=False).astype(
        int
    )
    df["is_cvt"] = t.str.contains(r"\bcvt\b", regex=True, na=False).astype(int)
    df["is_dct"] = t.str.contains(r"dual|dct|dual clutch", regex=True, na=False).astype(
        int
    )

    df["transmission_speeds"] = (
        df["transmission"]
        .str.extract(r"(\d+)\s*[- ]?\s*speed", flags=re.IGNORECASE)[0]
        .astype(float)
    )

    df["transmission_speeds_missing"] = df["transmission_speeds"].isna().astype(int)

    df["transmission_speeds"] = df["transmission_speeds"].fillna(0)

    df["has_dual_shift"] = t.str.contains(
        r"dual shift|dual clutch", regex=True, na=False
    ).astype(int)

    def simplify(x: str) -> str:
        x = str(x).lower().strip()

        if x in ["", "missing", "nan", "none"]:
            return "missing"
        if "manual" in x or "m/t" in x:
            return "manual"
        if "cvt" in x:
            return "cvt"
        if "dual" in x or "dct" in x or "dual clutch" in x:
            return "dct"
        if "automatic" in x or "a/t" in x:
            return "automatic"
        return "other"

    df["transmission_type"] = (
        df["transmission"].apply(simplify).fillna("missing").astype(str)
    )

    missing_mask = df["transmission_type"].eq("missing")
    df.loc[
        missing_mask,
        ["is_automatic", "is_manual", "is_cvt", "is_dct", "has_dual_shift"],
    ] = 0

    new_cols = [
        "transmission",
        "is_automatic",
        "is_manual",
        "is_cvt",
        "is_dct",
        "transmission_speeds",
        "transmission_speeds_missing",
        "has_dual_shift",
        "transmission_type",
    ]

    for col in new_cols:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("missing").astype(str)
        else:
            df[col] = df[col].fillna(0)

    return df


def add_engine_features(df: pd.DataFrame) -> pd.DataFrame:
    df["is_gasoline"] = (df["fuel_type"] == "Gasoline").astype(int)

    df["Electrical"] = (
        df["engine"].str.contains("Electric", case=False, na=False)
        | df["brand"].str.contains("Tesla", case=False, na=False)
        | df["model"].str.contains(
            "LONG RANGE|P100D|RECHARGE|PURE ELECTRIC", case=False, na=False
        )
    ).astype(int)

    engine_lower = df["engine"].str.lower()

    df["engine_hp"] = (
        df["engine"]
        .str.extract(r"(\d+\.?\d*)\s*HP", flags=re.IGNORECASE)[0]
        .astype(float)
    )

    # displacement in liters
    df["engine_liters"] = (
        df["engine"]
        .str.extract(r"(\d+\.?\d*)\s*L", flags=re.IGNORECASE)[0]
        .astype(float)
    )

    # cylinders
    df["engine_cylinders"] = (
        df["engine"]
        .str.extract(r"(\d+)\s*Cylinder", flags=re.IGNORECASE)[0]
        .astype(float)
    )

    # aspiration / tech
    df["is_turbo"] = engine_lower.str.contains(r"turbo").astype(int)
    df["is_twin_turbo"] = engine_lower.str.contains(r"twin turbo").astype(int)

    # derived engine ratios
    df["hp_per_liter"] = df["engine_hp"] / df["engine_liters"]
    df["hp_per_cylinder"] = df["engine_hp"] / df["engine_cylinders"]
    
    numeric_cols = [
        'engine_hp',
        'engine_liters',
        'engine_cylinders',
        'hp_per_liter',
        'hp_per_cylinder'
    ]

    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    return df


##########################


def add_brand_features_safe(final_df : pd.DataFrame) -> pd.DataFrame:
    
    df_train = final_df[final_df['_dataset'] == 'train'].copy()
    df_test = final_df[final_df['_dataset'] == 'test'].copy()

    brand_stats = (
        df_train.groupby("brand")
        .agg(
            train_brand_count=("price", "count"), train_brand_median=("price", "median")
        )
        .reset_index()
    )

    def apply_stats(target_df):
        res = target_df.merge(brand_stats, on="brand", how="left")

        # svakako nemamo ali neka ostane....
        res["train_brand_count"] = res["train_brand_count"].fillna(1)
        res["train_brand_median"] = res["train_brand_median"].fillna(
            df_train["price"].median()
        )

        res["brand_count_log"] = np.log1p(res["train_brand_count"])

        ratio = res["train_brand_median"] / res["train_brand_count"]
        res["brand_exclusivity_log"] = np.log1p(ratio)

        return res.drop(columns=["train_brand_count", "train_brand_median"])

    # Primena
    df_train_new = apply_stats(df_train)
    df_test_new = apply_stats(df_test)
    
    return pd.concat([df_train, df_test], axis=0, ignore_index=True)
 


def process_model_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    m = df["model"].fillna("MISSING").astype(str).str.upper()
    e = df["engine"].fillna("MISSING").astype(str).str.upper()

    def clean_doubles(text):
        words = text.split()
        return " ".join(dict.fromkeys(words))

    df["model_clean"] = m.apply(clean_doubles)

    perf_pattern = "SVR|ST|RED|BLACKWING|COMPETITION|S-MODEL|TYPE S|AMG|M-SPORT|RS"
    df["is_high_performance"] = m.str.contains(perf_pattern).astype(int)

    turbo_pattern = "TURBO|SUPERCHARGED|2\.0T|3\.0T|4\.0T|2\.4T|4\.2L"
    m_turbo = m.str.contains(turbo_pattern).astype(int)
    e_turbo = e.str.contains("TURBO|SUPER").astype(int)
    df["is_forced_induction"] = ((m_turbo == 1) | (e_turbo == 1)).astype(int)

    df["is_electric_long_range"] = m.str.contains(
        "LONG RANGE|P100D|RECHARGE|PURE ELECTRIC"
    ).astype(int)
    df["is_plugin_hybrid"] = (
        m.str.contains("PLUG-IN|PHEV|HYBRID") | e.str.contains("HYBRID")
    ).astype(int)

    special_keywords = (
        "EDITION|HERITAGE|SPECIAL|ANNIVERSARY|FINAL|CRAFTED|PMC|FIRST EDITION"
    )
    df["is_special_series"] = m.str.contains(special_keywords).astype(int)

    luxury_keywords = "AUTOBIOGRAPHY|WESTMINSTER|INSCRIPTION|LARIAT|ULTIMATE|PLATINUM|HIGH COUNTRY|DENALI"
    df["is_luxury_trim"] = m.str.contains(luxury_keywords).astype(int)

    df["is_extended_length"] = m.str.contains(
        r"\bLWB\b|\bL\b|ESV|\bXL\b|EXTENDED"
    ).astype(int)

    positive_cols = [
        "is_high_performance",
        "is_forced_induction",
        "is_electric_long_range",
        "is_plugin_hybrid",
        "is_special_series",
        "is_luxury_trim",
        "is_extended_length",
    ]

    df["positive_features_count"] = df[positive_cols].sum(axis=1)

    return df


def process_year_features(df):
    df = df.copy()

    current_year = 2024

    df["car_age"] = (current_year - df["model_year"]).clip(lower=0)

    df["car_age_squared"] = df["car_age"] ** 2

    df["log_car_age"] = np.log1p(df["car_age"])

    if "positive_features_count" in df.columns:
        df["luxury_score_per_age"] = df["positive_features_count"] / (df["car_age"] + 1)

    def get_car_era(age):
        if age <= 2:
            return "NEW"  # Miris novog auta
        if age <= 5:
            return "MODERN"  # Još uvek aktuelan model
        if age <= 12:
            return "USED"  # Standardni polovnjak
        if age <= 25:
            return "OLD"  # Na ivici rashoda
        return "VINTAGE"  # Potencijalni klasik

    df["car_era"] = df["car_age"].apply(get_car_era)

    return df


def process_mileage_features(df):
    df = df.copy()

    df["log_mileage"] = np.log1p(df["milage"])

    df["mileage_per_year"] = df["milage"] / (df["car_age"] + 1)

    df["mileage_squared"] = df["milage"] ** 2

    def mileage_category(m):
        if m < 15000:
            return "NEW_LIKE"
        if m < 60000:
            return "LOW_MILEAGE"
        if m < 150000:
            return "AVERAGE"
        if m < 250000:
            return "HIGH_MILEAGE"
        return "VERY_HIGH_MILEAGE"

    df["mileage_type"] = df["milage"].apply(mileage_category)

    df["is_high_usage"] = (df["mileage_per_year"] > 20000).astype(int)

    return df


def extract_primary_color(text):
    if pd.isna(text):
        return None

    text = str(text).lower()

    color_map = {
        'black': 'black', 'nero': 'black', 'onyx': 'black',
        'white': 'white', 'bianco': 'white', 'ice': 'white',
        'blue': 'blue', 'blu': 'blue',
        'red': 'red', 'rosso': 'red',
        'green': 'green',
        'yellow': 'yellow', 'giallo': 'yellow',
        'orange': 'orange',
        'grey': 'gray', 'gray': 'gray',
        'silver': 'silver',
        'brown': 'brown',
        'beige': 'beige',
        'gold': 'gold',
        'purple': 'purple',
    }

    for word in text.split():
        if word in color_map:
            return color_map[word]

    return 'other'


def color_family(color):
    if pd.isna(color):
        return None

    if color in ['black', 'gray', 'silver']:
        return 'neutral_dark'
    if color in ['white', 'beige']:
        return 'neutral_light'
    if color in ['red', 'orange', 'yellow', 'gold']:
        return 'warm'
    if color in ['blue', 'green', 'purple']:
        return 'cool'
    return 'other'


def is_metallic(text):
    if pd.isna(text):
        return 0
    return int('metallic' in str(text).lower())


def is_special_color(text):
    if pd.isna(text):
        return 0

    keywords = ['pearlescent', 'pearl', 'matte', 'satin']
    text = str(text).lower()
    return int(any(k in text for k in keywords))


def is_dark_color(color):
    if pd.isna(color):
        return 0
    return int(color in ['black', 'gray', 'blue', 'green', 'purple', 'brown'])


def extract_interior_color(text):
    if pd.isna(text):
        return None

    text = str(text).lower()

    color_map = {
        'black': 'black', 'nero': 'black', 'onyx': 'black',
        'white': 'white', 'bianco': 'white',
        'red': 'red', 'rosso': 'red',
        'beige': 'beige', 'tan': 'beige', 'cream': 'beige',
        'brown': 'brown', 'cuoio': 'brown',
        'grey': 'gray', 'gray': 'gray',
        'blue': 'blue', 'blu': 'blue',
        'green': 'green',
    }

    for word in text.split():
        if word in color_map:
            return color_map[word]

    return 'other'


def has_leather(text):
    if pd.isna(text):
        return 0

    text = str(text).lower()
    keywords = ['leather', 'alcantara', 'hide', 'cuoio']
    return int(any(k in text for k in keywords))


def create_color_features(df):
    df = df.copy()

    # eksterno
    df['primary_color'] = df['ext_col'].apply(extract_primary_color)
    df['color_family'] = df['primary_color'].apply(color_family)
    df['is_metallic'] = df['ext_col'].apply(is_metallic)
    df['is_special_color'] = df['ext_col'].apply(is_special_color)
    df['is_dark'] = df['primary_color'].apply(is_dark_color)

    # enterijer
    df['interior_color'] = df['int_col'].apply(extract_interior_color)
    df['has_leather'] = df['int_col'].apply(has_leather)

    return df
