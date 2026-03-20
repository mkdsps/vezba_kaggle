import pandas as pd
import re



def add_features(df : pd.DataFrame) -> pd.DataFrame:
    """
        vraca kompletan dataset sa dodatim features i izbrisanim ostalim stvarima ( spojeni train i test )
    """
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
    df = add_transmission_features(df)
    return df

def add_transmission_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1) raw transmission bez missing
    df['transmission'] = df['transmission'].fillna('missing').astype(str).str.strip()

    t = df['transmission'].str.lower()

    # 2) osnovni flagovi
    df['is_automatic'] = t.str.contains(r'automatic|\ba/t\b', regex=True, na=False).astype(int)
    df['is_manual'] = t.str.contains(r'manual|\bm/t\b', regex=True, na=False).astype(int)
    df['is_cvt'] = t.str.contains(r'\bcvt\b', regex=True, na=False).astype(int)
    df['is_dct'] = t.str.contains(r'dual|dct|dual clutch', regex=True, na=False).astype(int)

    # 3) broj brzina
    df['transmission_speeds'] = (
        df['transmission']
        .str.extract(r'(\d+)\s*[- ]?\s*speed', flags=re.IGNORECASE)[0]
        .astype(float)
    )

    # missing indikator pre fill
    df['transmission_speeds_missing'] = df['transmission_speeds'].isna().astype(int)

    # popuna broja brzina
    # bolje 0 nego NaN, jer znači "nije eksplicitno navedeno"
    df['transmission_speeds'] = df['transmission_speeds'].fillna(0)

    # 4) dodatni signal
    df['has_dual_shift'] = t.str.contains(r'dual shift|dual clutch', regex=True, na=False).astype(int)

    # 5) kategorija bez missing
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

    # 6) da nema kontradikcija / praznih slučajeva
    # ako je transmission_type missing, sve binary ostavi 0
    missing_mask = df['transmission_type'].eq('missing')
    df.loc[missing_mask, ['is_automatic', 'is_manual', 'is_cvt', 'is_dct', 'has_dual_shift']] = 0

    # ako ništa nije uhvaćeno, ostaje "other"
    # ne forsiramo automatic jer to ume da unese noise

    # 7) finalna sigurnosna provera: nema missing u novim kolonama
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