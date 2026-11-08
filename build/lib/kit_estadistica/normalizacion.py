import numpy as np
import pandas as pd

def normalizar_variable(x):
    """Normaliza un vector numérico individual entre 0 y 1."""
    minimo = min(x)
    maximo = max(x)
    if minimo == maximo:
        return [0.0 for _ in x]
    return [(val - minimo) / (maximo - minimo) for val in x]


def normalizar(df):
    """Normaliza todas las columnas numéricas de un DataFrame."""
    df_norm = pd.DataFrame(index=df.index)
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df_norm[col] = normalizar_variable(df[col].tolist())
        else:
            df_norm[col] = df[col]
    return df_norm


def estandarizar_variable(x):
    """Estandariza un vector numérico individual (media 0, desviación típica 1)."""
    media = np.mean(x)
    std = np.std(x)
    if std == 0:
        return [0.0 for _ in x]
    return [(val - media) / std for val in x]


def estandarizar(df):
    """Estandariza todas las columnas numéricas de un DataFrame."""
    df_estd = pd.DataFrame(index=df.index)
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df_estd[col] = estandarizar_variable(df[col].tolist())
        else:
            df_estd[col] = df[col]
    return df_estd
