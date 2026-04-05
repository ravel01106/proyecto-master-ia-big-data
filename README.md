# Proyecto final del master de FP de IA y Big Data

## Introducción del proyecto
Este proyecto simula un escenario real de un Data Scientist en una compañía de retail. El objetivo principal es resolver un problema de regresión para predecir el comportamiento de las ventas en un canal de e-commerce.

Se utiliza el dataset público "E-Commerce Data" de Kaggle el cual tiene las siguientes características:
* Transacciones de una tienda online del Reino Unido entre el 1 de diciembre de 2010 y el 9 de diciembre de 2011.
* Más de 500.000 filas con información sobre facturas, productos, cantidades y precios.
* Se encuentran en crudo y requieren una limpieza profunda debido a valores faltantes y devoluciones (cantidades negativas).

## Modelo de regresión
El reto específico de este módulo es predecir los beneficios diarios (valor total de ventas) para el periodo comprendido entre el 9 de noviembre de 2011 y el 9 de diciembre de 2011.
Para ello, se utiliza:
* Entrenamiento/Validación: Datos del 1 de diciembre de 2010 al 8 de noviembre de 2011.
* Métrica de Evaluación: RMSE (Root Mean Squared Error).


## Configuración técnica
Para ver los detalles sobre las librerías utilizadas, la estructura de carpetas y los pasos de instalación, consulta nuestra guía técnica:
**[Ver Guía de Configuración (SETUP.md)](./docs/setup.md)**