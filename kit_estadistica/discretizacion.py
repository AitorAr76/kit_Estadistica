import numpy as np
import pandas as pd
import math

def discretize(x, cut_points):
    """Discretiza una lista/array numérico en función de unos puntos de corte dados."""
    lag = [-np.inf]
    lag.extend(cut_points)
    lag.append(np.inf)
    x_discretized = []

    for elem in x:
        i = 1
        while i < len(lag) and elem > lag[i]:
            i += 1
        x_discretized.append(f"({lag[i-1]},{lag[i]})")

    return x_discretized


def discretizeEW(x, num_bins):
    """Discretización por Igual Anchura (Equal Width)."""
    x_min = min(x)
    x_max = max(x)
    if x_min == x_max:
        fr = 1.0
    else:
        fr = (x_max - x_min) / num_bins

    cut_points = [x_min + fr * i for i in range(1, num_bins)]
    x_discretized = discretize(x, cut_points)

    return x_discretized, cut_points


def discretizeEF(x, num_bins):
    """Discretización por Igual Frecuencia (Equal Frequency)."""
    orig_x = list(x)
    x_sorted = sorted(orig_x)

    n = len(x_sorted)
    lag = n % num_bins
    zenb = math.ceil(n / num_bins)

    cut_points = [x_sorted[zenb * elem - 1] for elem in range(1, lag + 1)]

    if lag < num_bins - 1:
        lag2 = [x_sorted[zenb * lag - 1 + max(zenb - 1, 1) * elem] for elem in range(1, num_bins - lag)]
        cut_points.extend(lag2)
        cut_points = cut_points[0:num_bins - 1]

    x_discretized = discretize(orig_x, cut_points)
        
    return x_discretized, cut_points


def discretize_dataframe(df, num_bins=3, metodo='EW'):
    """Discretiza todas las columnas numéricas de un DataFrame completo."""
    df_discretizado = pd.DataFrame(index=df.index)
    
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            x = df[col].tolist()
            if metodo == 'EW':
                x_disc, _ = discretizeEW(x, num_bins)
            elif metodo == 'EF':
                x_disc, _ = discretizeEF(x, num_bins)
            else:
                raise ValueError("Método no válido. Usa 'EW' o 'EF'.")
            df_discretizado[col] = x_disc
        else:
            df_discretizado[col] = df[col]
            
    return df_discretizado
