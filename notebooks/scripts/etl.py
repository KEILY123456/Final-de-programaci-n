import pandas as pd

def load_data(file_path):
    """Carga datos desde un archivo CSV."""
    return pd.read_csv(file_path)

def transform_data(df):
    """Realiza transformaciones en el dataframe."""
    # Aquí podrías agregar tus transformaciones
    df['new_column'] = df['existing_column'] * 2
    return df

def save_data(df, file_path):
    """Guarda el dataframe transformado en un archivo CSV."""
    df.to_csv(file_path, index=False)
