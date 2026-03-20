import pandas as pd


def clean_all(df_combined : pd.DataFrame) -> pd.DataFrame:
    """
        vraca ocisceni dataframe...
    """
    
    df_combined = clean_fuel_type(df_combined)

    df = df_combined.copy()
    return df

def clean_fuel_type(df_combined : pd.DataFrame) -> pd.DataFrame:

    df_combined = df_combined.copy()
    df_combined['fuel_type'] = df_combined['fuel_type'].replace('-', None)


    df_combined['Electrical'] = (
    df_combined['engine'].str.contains('Electric', case=False, na=False) |
    df_combined['brand'].str.contains('Tesla', case=False, na=False)
    ).astype(int)

    df_combined.loc[
    (df_combined['fuel_type'].isna()) & (df_combined['Electrical'] == 1),
    'fuel_type'
    ] = 'Electric'


    df_combined.loc[
    (df_combined['fuel_type'].isna()) & (df_combined['engine'].str.contains('Gasoline',case=False,na=False)),
    'fuel_type'
    ] = 'Gasoline'


    df_combined.loc[
    (df_combined['fuel_type'].isna()) & (df_combined['engine'].str.contains('Diesel',case=False,na=False)),
    'fuel_type'
    ] = 'Diesel'


    df_combined.loc[
    (df_combined['fuel_type'].isna()),
    'fuel_type'
    ] = 'Gasoline'

    df_combined['clean_title']= df_combined['clean_title'].fillna('Unknown')

    df_combined['accident']= df_combined['accident'].fillna('Unknown')
    
    return df_combined
