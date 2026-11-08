import matplotlib.pyplot as plt
import seaborn as sns
from .metricas import curva_roc_auc
from .correlacion import matriz_correlacion

def visualizar_curva_roc(x, y):
    """Dibuja la curva ROC y calcula el área bajo la curva (AUC)."""
    puntos, auc = curva_roc_auc(x, y)

    if not puntos:
        print('Aviso: No se pudo generar la curva ROC.')
        return auc

    fprs = [p[0] for p in puntos]
    tprs = [p[1] for p in puntos]

    plt.figure(figsize=(6, 6))
    plt.plot(
        fprs,
        tprs,
        color='darkorange',
        lw=2,
        label=f'Curva ROC (AUC = {auc:.4f})',
    )
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Azar')
    plt.xlabel('FPR (Tasa de Falsos Positivos)')
    plt.ylabel('TPR (Tasa de Verdaderos Positivos)')
    plt.title('Curva ROC y AUC')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.show()

    return auc


def visualizar_matriz_mi_corr(df):
    """Visualiza la matriz híbrida de correlación e información mutua en heatmap."""
    matriz = matriz_correlacion(df)
    plt.figure(figsize=(8, 6))
    sns.heatmap(matriz, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, vmin=0, vmax=1)
    plt.title('Matriz de Correlación / Información Mutua', fontsize=14)
    plt.tight_layout()
    plt.show()
