# Proyecto final del master de FP de IA y Big Data

## Introducción del proyecto
Este proyecto simula un escenario real de un Data Scientist en una compañía de retail. El objetivo principal es resolver un problema de regresión para predecir el comportamiento de las ventas en un canal de e-commerce.

Se utiliza el dataset público "E-Commerce Data" de Kaggle el cual tiene las siguientes características:
* Transacciones de una tienda online del Reino Unido entre el 1 de diciembre de 2010 y el 9 de diciembre de 2011.
* Más de 500.000 filas con información sobre facturas, productos, cantidades y precios.
* Se encuentran en crudo y requieren una limpieza profunda debido a valores faltantes y devoluciones (cantidades negativas).

## Modelo de regresión
El reto específico de este módulo es predecir los beneficios diarios (valor total de ventas) para el periodo comprendido entre el 9 de noviembre de 2011 y el 9 de diciembre de 2011.
### Flujo de Trabajo
* **Limpieza Temporal:** Gestión de registros negativos (devoluciones) y depuración de precios unitarios anómalos.
* **Feature Engineering:** Creación de variables de retardo (*Lags*), medias móviles de 7 días y codificación de estacionalidad (fines de semana, días de la semana).
* **Modelado Comparativo:**
   - **Baseline:** ARIMA (Estadística clásica de series temporales).
   - **Avanzado:** Random Forest Regressor (Ensamble de aprendizaje supervisado).

### Métricas de Evaluación
* **RMSE** (Root Mean Squared Error): Para penalizar desviaciones grandes.
* **MAE** (Mean Absolute Error): Error medio en términos monetarios.
* **MAPE** (Mean Absolute Percentage Error): *Métrica clave, reducida del 30.15% (ARIMA) al 15.36% (Random Forest).*


## Configuración técnica
Para ver los detalles sobre las librerías utilizadas, la estructura de carpetas y los pasos de instalación, consulta nuestra guía técnica:
**[Ver Guía de Configuración (SETUP.md)](./docs/setup.md)**