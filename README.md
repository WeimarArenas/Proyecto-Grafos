# Backward Oversmoothing y PairNorm en una GNN profunda

Proyecto final del curso **Aprendizaje automático en grafos**, Universidad de Antioquia.

**Integrantes:** Manuel Quistial · Weimar Arenas

Artículo de referencia: N. Keriven, [*Backward Oversmoothing: why is it hard to train deep Graph Neural Networks?*](https://openreview.net/pdf?id=dx53sSG8py), ICML 2026.
Artículo de apoyo: PairNorm, L. Zhao y L. Akoglu, [*PairNorm: Tackling Oversmoothing in GNNs*](https://openreview.net/forum?id=rkecl1rtwB), ICLR 2020.

## Contenido

| Archivo | Descripción |
|---|---|
| [`Keriven_PairNorm.ipynb`](Keriven_PairNorm.ipynb) | Notebook del experimento |
| [`Presentacion_Backward_Oversmoothing.pptx`](Presentacion_Backward_Oversmoothing.pptx) | Presentación del proyecto |

**Pregunta.** ¿Incorporar PairNorm en las capas ocultas de una GNN de 100 capas reduce el *backward oversmoothing* y permite alcanzar una menor pérdida de entrenamiento que la misma red sin normalización?

**Experimento.** Cora (componente conexa mayor, 2.485 nodos), $P=D^{-1}A$, 100 capas, Leaky ReLU, salida lineal, pérdida MSE del artículo, pesos iniciales $I+0.001N$ y Adam. Hay tres configuraciones (C1, C2 y C3) con cuatro semillas y dos condiciones, sin y con PairNorm, que dan **24 entrenamientos** de 1.000 épocas. El par de modelos de cada semilla parte de pesos idénticos.

**Resultado.** La hipótesis queda parcialmente respaldada:

- La red sin PairNorm terminó con menor pérdida en los 12 pares.
- PairNorm aumentó $E_\pi(B)$ en las capas intermedias, pero también aumentó la norma de la señal backward.

Las tablas y figuras están en las secciones 9 a 11 del notebook.

## Ejecutar en Google Colab

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/WeimarArenas/Proyecto-Grafos/blob/main/Keriven_PairNorm.ipynb)

No hace falta instalar nada localmente, la primera celda instala las dependencias y la celda del dataset Cora descarga los datos.

1. **Abrir el notebook** con el botón anterior, colab abre una copia.
2. **(Opcional) Activar GPU** en *Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU T4*. El código detecta la GPU automáticamente, la ejecución guardada en el notebook se hizo en CPU y tardó alrededor de **1 h 30 min**.
3. **Instalar dependencias.** Ejecutar solo la celda de la sección 2 (`%pip install … torch-geometric==2.8.1 …`).
4. **Reiniciar la sesión** con *Entorno de ejecución → Reiniciar sesión*, este paso es necesario para que `torch_geometric` se importe correctamente.
5. **Ejecutar todo** con *Entorno de ejecución → Ejecutar todo*, la celda de instalación vuelve a correr, pero termina en segundos.
6. **Autorizar Google Drive** cuando Colab lo pida, en la celda de la sección 3, los resultados y checkpoints se guardan en `MyDrive/Keriven_PairNorm`.
7. Al terminar, el notebook **descarga un ZIP** (`Keriven_PairNorm_Resultados.zip`) con los CSV y las figuras.

### Si no se quiere conectar con Google Drive

En la celda de la sección 3, comentar la línea `drive.mount` y usar la ruta local que ya está escrita como alternativa:

```python
# from google.colab import drive; drive.mount('/content/drive')
# OUTPUT_DIR = Path('/content/drive/MyDrive/Keriven_PairNorm')
OUTPUT_DIR = Path('Keriven_PairNorm').resolve()
```

Así los resultados quedan en `/content/Keriven_PairNorm` y en el ZIP final, pero se pierden al cerrar la sesión de Colab.

### Si la sesión se desconecta

Cada entrenamiento guarda un checkpoint cada 100 épocas, si Colab se desconecta, repita los pasos 4 y 5:

- Los entrenamientos ya completados no se repiten;
- El entrenamiento interrumpido continúa desde su último checkpoint.

Para empezar desde cero, borre la carpeta `MyDrive/Keriven_PairNorm`.

### Archivos que genera

| Archivo | Contenido |
|---|---|
| `runs/<config>_s<semilla>_<condición>/` | Pérdida por época (`historias.csv`), métricas por capa (`perfiles.csv`) y resumen (`resumen.json`) de cada entrenamiento. |
| `comparacion_pareada.csv` | Pérdida final con y sin PairNorm para cada configuración y semilla. |
| `resumen_backward_central.csv` | $E_\pi(B)$ y $\lVert B\rVert_F$ promediados en las capas 34–67, época 1.000. |
| `C1/C2/C3_backward.png`, `C1/C2/C3_perdida.png` | Perfiles backward por capa y curvas de pérdida. |
| `manifest_protocolo_C1_C2_C3.json` | Configuración, versiones y datos del experimento. |

Versiones registradas en la ejecución guardada: Python 3.13, PyTorch 2.14.1 y PyTorch Geometric 2.8.1.

## Referencias

1. Keriven, N. (2026). *Backward Oversmoothing: why is it hard to train deep Graph Neural Networks?* ICML, PMLR 306, 56740–56763.
2. Zhao, L. y Akoglu, L. (2020). *PairNorm: Tackling Oversmoothing in GNNs.* ICLR.
3. Yang, Z., Cohen, W. W. y Salakhutdinov, R. (2016). *Revisiting Semi-Supervised Learning with Graph Embeddings.* ICML, PMLR 48, 40–48.
