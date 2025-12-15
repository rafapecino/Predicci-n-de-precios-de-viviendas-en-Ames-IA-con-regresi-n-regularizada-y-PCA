# 🏡 Predicción de Precios de Viviendas con Regresión Regularizada

Este repositorio presenta un proyecto de ciencia de datos de principio a fin para predecir los precios de las viviendas utilizando el [dataset de Ames, Iowa](https://www.kaggle.com/c/house-prices-advanced-regression-techniques). El objetivo es demostrar un flujo de trabajo robusto que incluye análisis exploratorio, preprocesamiento avanzado, modelado y evaluación, con un enfoque en la interpretabilidad y el rendimiento predictivo.

![Heatmap de Correlaciones](plots/img_02_heatmap.png)

## 🎯 Objetivo del Proyecto

El principal desafío de este dataset es su alta dimensionalidad (80 variables) y la presencia de multicolinealidad. El objetivo es construir un modelo de regresión que:
1.  **Maneje la complejidad**: Gestione un gran número de características sin sobreajuste.
2.  **Sea Interpretable**: Identifique qué factores son más influyentes en la determinación del precio.
3.  **Sea Preciso**: Minimice el error de predicción en datos no vistos.

## 🛠️ Flujo de Trabajo y Tecnologías

El análisis se implementó íntegramente en **R**, utilizando librerías como `tidyverse`, `glmnet`, y `caret`.

1.  **Análisis Exploratorio de Datos (EDA)**:
    *   Visualización de la distribución de `SalePrice` y corrección de su asimetría mediante **transformación logarítmica**.
    *   Análisis de correlaciones para detectar multicolinealidad.
    *   Estudio de la relación entre características clave (`OverallQual`, `GrLivArea`) y el precio.

2.  **Preprocesamiento de Datos**:
    *   Imputación de valores ausentes basándose en la naturaleza de cada variable (ej. "None" para características ausentes).
    *   **Codificación One-Hot** para variables categóricas.
    *   **Estandarización Z-score** de todas las variables numéricas para que sean comparables.

3.  **Modelado y Evaluación**:
    *   **Análisis de Componentes Principales (PCA)**: Utilizado como herramienta diagnóstica, reveló que se necesitan más de 120 componentes para explicar el 95% de la varianza.
    *   **Regresión Ridge (L2)**: Entrenado como baseline para manejar multicolinealidad.
    *   **Regresión Lasso (L1)**: Entrenado para realizar selección automática de características y mejorar la interpretabilidad.
    *   **Validación Cruzada (10-fold)**: Aplicada para encontrar el hiperparámetro de regularización óptimo ($\lambda$) para ambos modelos.

## 📊 Resultados y Conclusiones

Se compararon los modelos en un conjunto de prueba (30% de los datos).

| Modelo | RMSE (USD) | MAE (USD) | R² | Variables Seleccionadas |
| :--- | :--- | :--- | :--- | :--- |
| Ridge | \$22,400 | \$14,800 | 0.91 | 80 de 80 |
| **Lasso** | **\$22,581** | **\$14,249** | **0.92** | **46 de 80** |

![Predicción vs Realidad](plots/img_05_pred_vs_real.png)

**Conclusión Final**: El modelo **Lasso** es superior. Aunque su RMSE es marginalmente más alto que el de Ridge, ofrece un **mejor R²** y un **MAE más bajo**. Su principal ventaja es que **simplifica el modelo de 80 a solo 46 variables**, proporcionando una solución más parsimoniosa e interpretable sin sacrificar rendimiento predictivo.

## 📂 Estructura del Repositorio

.
├── data/
│ ├── train.csv
│ └── data_description.txt
├── plots/
│ ├── img_01_distribucion_precio.png
│ ├── img_02_heatmap.png
│ └── ... (todas las visualizaciones)
├── script.R
├── informe.pdf
└── README.md
## 🚀 Cómo Reproducir el Análisis

1.  Clona este repositorio:
    ```
    git clone https://github.com/tu-usuario/tu-repositorio.git
    cd tu-repositorio
    ```
2.  Asegúrate de tener las librerías de R necesarias instaladas (`tidyverse`, `glmnet`, `caret`).
3.  Ejecuta el script principal en tu entorno de R:
    ```
    source("script.R")
    ```
    El script se encargará de cargar los datos, procesarlos, entrenar los modelos, generar las métricas y guardar las visualizaciones en la carpeta `plots/`.

---
*Este proyecto demuestra habilidades en análisis estadístico, modelado predictivo y comunicación técnica de resultados.*
