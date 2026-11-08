import pandas as pd
import math
from .discretizacion import discretize_dataframe

def informacion_mutua(x, y):
    """Calcula la Información Mutua entre dos variables discretas o discretizadas."""
    df_temp = pd.DataFrame({'x': x, 'y': y})
    n = len(df_temp)
    contingency = pd.crosstab(df_temp['x'], df_temp['y'])
    
    mi = 0.0
    p_x = contingency.sum(axis=1) / n
    p_y = contingency.sum(axis=0) / n
    
    for i in range(contingency.shape[0]):
        for j in range(contingency.shape[1]):
            p_xy = contingency.iloc[i, j] / n
            if p_xy > 0:
                mi += p_xy * math.log2(p_xy / (p_x.iloc[i] * p_y.iloc[j]))
    return mi


def matriz_correlacion(df, num_bins_disc=3):
    """Calcula la matriz por pares considerando el tipo de variable."""
    cols = df.columns
    n_cols = len(cols)
    matriz_resultado = pd.DataFrame(index=cols, columns=cols, dtype=float)
    
    df_disc = discretize_dataframe(df, num_bins=num_bins_disc, metodo='EW')
    
    for i in range(n_cols):
        for j in range(n_cols):
            col1 = cols[i]
            col2 = cols[j]
            
            if i == j:
                matriz_resultado.loc[col1, col2] = 1.0
            elif pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]) and df[col1].nunique() > 10 and df[col2].nunique() > 10:
                matriz_resultado.loc[col1, col2] = df[col1].corr(df[col2])
            else:
                mi = informacion_mutua(df_disc[col1], df_disc[col2])
                matriz_resultado.loc[col1, col2] = mi
                
    return matriz_resultado.astype(float)
