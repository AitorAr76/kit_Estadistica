from .discretizacion import discretize, discretizeEW, discretizeEF, discretize_dataframe
from .normalizacion import normalizar_variable, normalizar, estandarizar_variable, estandarizar
from .metricas import varianza, entropy, curva_roc_auc, calcular_metricas_dataset, filtrar_variables
from .correlacion import informacion_mutua, matriz_correlacion
from .visualizacion import visualizar_curva_roc, visualizar_matriz_mi_corr

__version__ = "0.1.0"
