import pandas as pd
import re



def add_features(df : pd.DataFrame) -> pd.DataFrame:
    """
        vraca kompletan dataset sa dodatim features i izbrisanim ostalim stvarima ( spojeni train i test )
    """
    df = add_engine_features(df)
    df = add_transmission_features(df)  
    return df  

def add_transmission_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df['transmission'] = df['transmission'].fillna('missing').astype(str).str.strip()

    t = df['transmission'].str.lower()

    df['is_automatic'] = t.str.contains(r'automatic|\ba/t\b', regex=True, na=False).astype(int)
    df['is_manual'] = t.str.contains(r'manual|\bm/t\b', regex=True, na=False).astype(int)
    df['is_cvt'] = t.str.contains(r'\bcvt\b', regex=True, na=False).astype(int)
    df['is_dct'] = t.str.contains(r'dual|dct|dual clutch', regex=True, na=False).astype(int)

    df['transmission_speeds'] = (
        df['transmission']
        .str.extract(r'(\d+)\s*[- ]?\s*speed', flags=re.IGNORECASE)[0]
        .astype(float)
    )

    df['transmission_speeds_missing'] = df['transmission_speeds'].isna().astype(int)

    df['transmission_speeds'] = df['transmission_speeds'].fillna(0)

    df['has_dual_shift'] = t.str.contains(r'dual shift|dual clutch', regex=True, na=False).astype(int)

    def simplify(x: str) -> str:
        x = str(x).lower().strip()

        if x in ['', 'missing', 'nan', 'none']:
            return 'missing'
        if 'manual' in x or 'm/t' in x:
            return 'manual'
        if 'cvt' in x:
            return 'cvt'
        if 'dual' in x or 'dct' in x or 'dual clutch' in x:
            return 'dct'
        if 'automatic' in x or 'a/t' in x:
            return 'automatic'
        return 'other'

    df['transmission_type'] = df['transmission'].apply(simplify).fillna('missing').astype(str)

    missing_mask = df['transmission_type'].eq('missing')
    df.loc[missing_mask, ['is_automatic', 'is_manual', 'is_cvt', 'is_dct', 'has_dual_shift']] = 0

    new_cols = [
        'transmission',
        'is_automatic',
        'is_manual',
        'is_cvt',
        'is_dct',
        'transmission_speeds',
        'transmission_speeds_missing',
        'has_dual_shift',
        'transmission_type'
    ]

    for col in new_cols:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('missing').astype(str)
        else:
            df[col] = df[col].fillna(0)

    return df

def add_engine_features(df: pd.DataFrame) -> pd.DataFrame:
    df['is_gasoline'] = (df['fuel_type'] == 'Gasoline').astype(int)

    df['Electrical'] = (
    df['engine'].str.contains('Electric', case=False, na=False) |
    df['brand'].str.contains('Tesla', case=False, na=False) |
    df['model'].str.contains('LONG RANGE|P100D|RECHARGE|PURE ELECTRIC', case=False, na=False)
    ).astype(int)
    

    engine_lower = df['engine'].str.lower()

    df['engine_hp'] = (
        df['engine']
        .str.extract(r'(\d+\.?\d*)\s*HP', flags=re.IGNORECASE)[0]
        .astype(float)
    )

    # displacement in liters
    df['engine_liters'] = (
        df['engine']
        .str.extract(r'(\d+\.?\d*)\s*L', flags=re.IGNORECASE)[0]
        .astype(float)
    )

    # cylinders
    df['engine_cylinders'] = (
        df['engine']
        .str.extract(r'(\d+)\s*Cylinder', flags=re.IGNORECASE)[0]
        .astype(float)
    )


    # aspiration / tech
    df['is_turbo'] = engine_lower.str.contains(r'turbo').astype(int)
    df['is_twin_turbo'] = engine_lower.str.contains(r'twin turbo').astype(int)


    # derived engine ratios
    df['hp_per_liter'] = df['engine_hp'] / df['engine_liters']
    df['hp_per_cylinder'] = df['engine_hp'] / df['engine_cylinders']
    
    return df
