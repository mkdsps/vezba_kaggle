import pandas as pd




def add_features(df_cleaned : pd.DataFrame) -> pd.DataFrame:
    """
        vraca kompletan dataset sa dodatim features i izbrisanim ostalim stvarima ( spojeni train i test )
    """
    df_cleaned['is_gasoline'] = (df_cleaned['fuel_type'] == 'Gasoline').astype(int)

    df_cleaned['Electrical'] = (
    df_cleaned['engine'].str.contains('Electric', case=False, na=False) |
    df_cleaned['brand'].str.contains('Tesla', case=False, na=False) |
    df_cleaned['model'].str.contains('LONG RANGE|P100D|RECHARGE|PURE ELECTRIC', case=False, na=False)
    ).astype(int)

    return df_cleaned
