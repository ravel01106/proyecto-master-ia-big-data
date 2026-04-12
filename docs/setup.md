# Guía de Configuración Técnica y Estructura
Este documento detalla la organización del repositorio y los pasos necesarios para replicar el entorno de desarrollo y la ejecución de los modelos de predicción.

# Estructura del Repositorio
```text
PROYECTO-MASTER-IA-BIG-DATA/
├── .venv/                          # Entorno virtual de Python
├── data/                           # Almacenamiento de datasets en distintas etapas
│   ├── interim/                    # Datos tras limpieza inicial (data_sanitized.csv)
│   ├── processed/                  # Dataset final dividido (test_data.csv, train_data.csv)
│   └── raw/                        # Dataset original sin procesar (data.csv)
├── docs/
│   └── setup.md                    # Documentación de configuración técnica
├── models/                         # Modelos entrenados exportados en formato binario
│   ├── arima_sales_model_v1.pkl
│   └── random_forest_v1.pkl
├── notebooks/regression_proyect/   # Flujo de trabajo segmentado por responsabilidades
│   ├── analysis/                   # Análisis Exploratorio de Datos (eda_data.ipynb)
│   ├── clean/                      # Saneamiento y tratamiento de nulos (clean_data.ipynb)
│   ├── draft_models/               # Entrenamiento de modelos (arima_model.ipynb, random_forest_model.ipynb)
│   └── preprocesing/               # Ingeniería de variables (preprocesing_data.ipynb)
├── reports/                        # Documentación y reportes de resultados
│   └── reporte_modelo_regresión_v0.1.odt
├── .gitignore                      # Exclusiones para control de versiones
├── README.md                       # Descripción general del proyecto
└── requirements.txt                # Librerías y dependencias necesarias
```

# Tecnologías y Dependencias
El proyecto utiliza Python 3.12+ y las siguientes librerías principales:
   - Procesado y Estadística: Pandas, NumPy, Statsmodels (ARIMA).
   - Aprendizaje Automático: Scikit-Learn (Random Forest, Escalado).
   - Visualización: Matplotlib, Seaborn.
   - Persistencia: Joblib.

# Instalación y Uso

## Preparación del Entorno

Es recomendable utilizar el entorno virtual incluido para garantizar la compatibilidad de las librerías:
Bash

### Activar el entorno virtual (Windows)
```text
.venv\Scripts\activate
```
### Instalación de dependencias
```text
pip install -r requirements.txt
```

# Flujo de Trabajo (Pipeline)

Para replicar el experimento completo, se debe seguir el orden de ejecución de los notebooks en notebooks/regression_proyect/:

   - analysis/eda_data.ipynb: Inspección visual y estadística inicial.

   - clean/clean_data.ipynb: Limpieza de nulos y duplicados. Genera data_sanitized.csv.

   - preprocesing/preprocesing_data.ipynb: Creación de lags y medias móviles. Genera los archivos en data/processed/.

   - draft_models/: Ejecución de los notebooks de entrenamiento para generar las predicciones finales.

# Notas de Mantenimiento
- Los modelos se guardan en la carpeta /models con versionado en el nombre (v1).
- Los datos de entrada originales deben situarse en /data/raw/data.csv para iniciar el pipeline