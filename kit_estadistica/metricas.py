import numpy as np
import pandas as pd
import math

def varianza(x):
    """Calcula la varianza de una lista o array numérico."""
    n = len(x)
    if n == 0:
        return 0.0
    media = sum(x) / n
    suma_cuadrados = sum([(elem - media)**2 for elem in x])
    return suma_cuadrados / n


def entropy(x):
    """Calcula la entropía de Shannon de una lista o variable discreta."""
    n = len(x)
    if n == 0:
        return 0.0

    kont = dict()
    for elem in x:
        if elem in kont:
            kont[elem] += 1
        else:
            kont[elem] = 1

    h = sum([(elem/n) * math.log2(elem/n) for elem in kont.values()])
    return h


def curva_roc_auc(x, y):
    """Cálculo de ROC y AUC respetando la dirección natural de la variable."""
    df_temp = pd.DataFrame({'var': x, 'label': y}).dropna().reset_index(drop=True)
    if len(df_temp) == 0:
        return [], 0.5

    x_vals = df_temp['var'].values
    y_raw = df_temp['label'].values

    if y_raw.dtype == object or isinstance(y_raw[0], str):
        y_bool = np.array([str(val).lower() in ['1', 'true', 'yes', 'pos', 'positivo', '1.0'] for val in y_raw])
    else:
        y_bool = y_raw.astype(bool)

    total_positivos = y_bool.sum()
    total_negativos = (~y_bool).sum()

    if total_positivos == 0 or total_negativos == 0:
        return [], 0.5

    desc_indices = np.argsort(-x_vals)
    x_sorted = x_vals[desc_indices]
    y_sorted = y_bool[desc_indices]

    tps = []
    fps = []
    current_tp = 0
    current_fp = 0
    
    for val in y_sorted:
        if val:
            current_tp += 1
        else:
            current_fp += 1
        tps.append(current_tp)
        fps.append(current_fp)
        
    tps = np.array(tps)
    fps = np.array(fps)

    puntos = [(0.0, 0.0)]
    
    for i in range(len(x_sorted)):
        if i == len(x_sorted) - 1 or x_sorted[i] != x_sorted[i + 1]:
            tpr = tps[i] / total_positivos
            fpr = fps[i] / total_negativos
            puntos.append((fpr, tpr))

    if puntos[-1] != (1.0, 1.0):
        puntos.append((1.0, 1.0))

    area = 0.0
    for i in range(1, len(puntos)):
        fpr1, tpr1 = puntos[i-1]
        fpr2, tpr2 = puntos[i]
        area += (fpr2 - fpr1) * (tpr1 + tpr2) / 2.0

    return puntos, area


def calcular_metricas_dataset(df, clase_binaria=None):
    """Calcula las métricas para los atributos de un dataset reconociendo su tipo."""
    resultados = {}
    
    for col in df.columns:
        if clase_binaria and col == clase_binaria:
            continue
            
        serie = df[col]
        if pd.api.types.is_numeric_dtype(serie):
            if serie.nunique() > 10:
                var_val = varianza(serie.tolist())
                auc_val = None
                if clase_binaria and clase_binaria in df.columns:
                    _, auc_val = curva_roc_auc(serie, df[clase_binaria])
                resultados[col] = {'tipo': 'continua', 'varianza': var_val, 'auc': auc_val}
            else:
                ent_val = entropy(serie.tolist())
                resultados[col] = {'tipo': 'discreta_numerica', 'entropia': ent_val}
        else:
            ent_val = entropy(serie.tolist())
            resultados[col] = {'tipo': 'categorica', 'entropia': ent_val}
            
    return pd.DataFrame.from_dict(resultados, orient='index')


def filtrar_variables(df, metrica, umbral, condicion='mayor', clase_binaria=None):
    """Filtra un dataset devolviendo uno nuevo donde las variables cumplen un requisito."""
    df_metricas = calcular_metricas_dataset(df, clase_binaria)
    cols_validas = []
    
    for col in df.columns:
        if clase_binaria and col == clase_binaria:
            cols_validas.append(col)
            continue
            
        if col in df_metricas.index:
            val = df_metricas.loc[col, metrica]
            if pd.isna(val):
                continue
            if condicion == 'mayor' and val > umbral:
                cols_validas.append(col)
            elif condicion == 'menor' and val < umbral:
                cols_validas.append(col)
                
    return df[cols_validas]
