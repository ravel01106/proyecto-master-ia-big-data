# Guía de Configuración Técnica y Estructura

En este documento se detallan las herramientas y la organización necesarias para replicar los experimentos de regresión del proyecto.

## Tecnologías y Librerías
El proyecto se desarrolla íntegramente en **Python**. Se utilizan las siguientes librerías especializadas:
* **Procesado de datos**: `Pandas` para la manipulación de datasets tabulares y procesos ETL simplificados.
* **Visualización**: `Matplotlib` y `Seaborn` para el análisis exploratorio y gráficos de calidad para la memoria.
* **Machine Learning**: `Scikit-Learn` para escalado, codificación y modelos como Random Forest.
* **Series Temporales**: `Statsmodels` (ARIMA) y `Prophet` para capturar tendencias y estacionalidad.
* **Modelos Avanzados**: `XGBoost` y `CatBoost` para algoritmos de boosting.
* **Deep Learning**: `TensorFlow/Keras` para la implementación de redes recurrentes **LSTM**.

## Estructura del Repositorio
Para facilitar la colaboración en GitHub, mantenemos una estructura organizada:
* `data/raw/`: Dataset original "E-Commerce Data" sin procesar.
* `data/processed/`: Datos tras la limpieza, feature engineering y creación de la variable "Ventas".
* `notebooks/borradores/`: Archivos `.ipynb` individuales para que cada miembro del equipo realice sus pruebas.
* `reports/`: Documentación, borradores y la memoria final en PDF.

## Instalación y Uso
1. **Clonar el repositorio** en tu entorno local de VS Code.
2. **Descargar el dataset** de Kaggle y guardarlo en la ruta `data/raw/`.
3. **Instalar dependencias**:
   ```bash
   pip install -r requirements.txt