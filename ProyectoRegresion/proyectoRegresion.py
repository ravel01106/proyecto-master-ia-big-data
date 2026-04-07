import pandas as pd

# 1. CARGA DEL DATASET

df = pd.read_csv(
    'contenidoCSV/data.csv',
    encoding='latin-1'
)

print("=== CARGA DEL DATASET ===")
print(f"Filas: {df.shape[0]} | Columnas: {df.shape[1]}")

print("\nPrimeras filas:")
print(df.head())

print("\nTipos de datos:")
print(df.dtypes)

# 2. ANÁLISIS EXPLORATORIO - ENTENDIMIENTO DE LOS DATOS

print("\n\n=== 2. ENTENDIMIENTO DE LOS DATOS ===")

# 2.1 Entendimiento de los datos

print("\n--- 2.1.1 Dimensiones y tipos de datos ---")
print(f"Filas: {df.shape[0]} | Columnas: {df.shape[1]}")
print("\nTipos de datos por columna:")
print(df.dtypes)

print("\n--- 2.1.2 Primeras filas del dataset ---")
print(df.head(10))

print("\n--- 2.1.3 Últimas filas del dataset ---")
print(df.tail(10))

print("\n--- 2.1.4 Valores únicos por columna ---")
for col in df.columns:
    print(f"  {col}: {df[col].nunique()} valores únicos")

print("\n--- 2.1.5 Muestra de valores únicos (columnas categóricas) ---")
columnas_categoricas = ['InvoiceNo', 'StockCode', 'Description', 'Country']
for col in columnas_categoricas:
    muestra = df[col].dropna().unique()[:10]
    print(f"\n  {col} (primeros 10 únicos):")
    print(f"  {muestra}")

print("\n--- 2.1.6 Estadísticas descriptivas (columnas numéricas) ---")
print(df.describe())

print("\n--- 2.1.7 Transacciones con Quantity <= 0 ---")
trans_qty_negativa = df[df['Quantity'] <= 0]
print(f"  Total filas con Quantity <= 0: {len(trans_qty_negativa)}")
print(f"  % sobre total: {len(trans_qty_negativa) / len(df) * 100:.2f}%")
print(trans_qty_negativa[['InvoiceNo', 'StockCode', 'Description', 'Quantity', 'UnitPrice']].head(10))

print("\n--- 2.1.8 Transacciones con UnitPrice <= 0 ---")
trans_price_negativa = df[df['UnitPrice'] <= 0]
print(f"  Total filas con UnitPrice <= 0: {len(trans_price_negativa)}")
print(f"  % sobre total: {len(trans_price_negativa) / len(df) * 100:.2f}%")
print(trans_price_negativa[['InvoiceNo', 'StockCode', 'Description', 'Quantity', 'UnitPrice']].head(10))

# 2.2 BÚSQUEDA DE VALORES FALTANTES, DUPLICADOS Y ERRÓNEOS

print("\n\n=== 2.2 VALORES FALTANTES, DUPLICADOS Y ERRÓNEOS ===")

print("\n--- 2.2.1 Valores faltantes (NaN) por columna ---")
nulos = df.isnull().sum()
nulos_pct = nulos / len(df) * 100
resumen_nulos = pd.DataFrame({
    'Nulos': nulos,
    '% sobre total': nulos_pct.round(2)
})
print(resumen_nulos[resumen_nulos['Nulos'] > 0])

print("\n--- 2.2.2 Filas sin CustomerID ---")
sin_cliente = df[df['CustomerID'].isnull()]
print(f"  Total filas sin CustomerID: {len(sin_cliente)}")
print(f"  % sobre total: {len(sin_cliente) / len(df) * 100:.2f}%")
print(f"\n  Distribución por país (top 10):")
print(sin_cliente['Country'].value_counts().head(10))

print("\n--- 2.2.3 Filas sin Description ---")
sin_descripcion = df[df['Description'].isnull()]
print(f"  Total filas sin Description: {len(sin_descripcion)}")
print(f"  % sobre total: {len(sin_descripcion) / len(df) * 100:.2f}%")
print(sin_descripcion[['InvoiceNo', 'StockCode', 'Quantity', 'UnitPrice', 'CustomerID']].head(10))

print("\n--- 2.2.4 Filas sin CustomerID y sin Description ---")
sin_ambos = df[df['CustomerID'].isnull() & df['Description'].isnull()]
print(f"  Total filas sin ambos: {len(sin_ambos)}")

print("\n--- 2.2.5 Filas duplicadas (exactas) ---")
duplicados = df.duplicated()
print(f"  Total filas duplicadas: {duplicados.sum()}")
print(f"  % sobre total: {duplicados.sum() / len(df) * 100:.2f}%")
print(df[duplicados].head(10))

print("\n--- 2.2.6 Formato de InvoiceDate (actualmente string) ---")
print(f"  Muestra de valores de InvoiceDate:")
print(df['InvoiceDate'].value_counts().head(10))
print(f"\n  Rango de fechas (como texto, orden lexicográfico):")
print(f"  Min: {df['InvoiceDate'].min()}")
print(f"  Max: {df['InvoiceDate'].max()}")

print("\n--- 2.2.7 StockCodes no estándar (posibles errores) ---")
# Los códigos estándar son alfanuméricos de 5-6 caracteres
# Códigos especiales conocidos: POST, D, C2, M, BANK CHARGES, PADS, DOT...
stock_no_estandar = df[~df['StockCode'].str.match(r'^[0-9]{5}[A-Za-z]?$', na=False)]
print(f"  Total filas con StockCode no estándar: {len(stock_no_estandar)}")
print(f"  Tipos de StockCodes no estándar (top 15):")
print(stock_no_estandar['StockCode'].value_counts().head(15))

print("\n--- 2.2.8 Distribucion de Country (value_counts) ---")
print(df['Country'].value_counts())

# 2.3 BÚSQUEDA DE OUTLIERS

print("\n\n=== 2.3 BUSQUEDA DE OUTLIERS ===")

print("\n--- 2.3.1 Outliers en Quantity (método IQR) ---")
Q1_qty = df['Quantity'].quantile(0.25)
Q3_qty = df['Quantity'].quantile(0.75)
IQR_qty = Q3_qty - Q1_qty
limite_inf_qty = Q1_qty - 1.5 * IQR_qty
limite_sup_qty = Q3_qty + 1.5 * IQR_qty
outliers_qty = df[(df['Quantity'] < limite_inf_qty) | (df['Quantity'] > limite_sup_qty)]
print(f"  Q1: {Q1_qty} | Q3: {Q3_qty} | IQR: {IQR_qty}")
print(f"  Limite inferior: {limite_inf_qty} | Limite superior: {limite_sup_qty}")
print(f"  Total outliers en Quantity: {len(outliers_qty)} ({len(outliers_qty) / len(df) * 100:.2f}%)")
print(f"  Valores extremos (top 5 mayores):")
print(df.nlargest(5, 'Quantity')[['InvoiceNo', 'StockCode', 'Description', 'Quantity', 'UnitPrice']])
print(f"  Valores extremos (top 5 menores):")
print(df.nsmallest(5, 'Quantity')[['InvoiceNo', 'StockCode', 'Description', 'Quantity', 'UnitPrice']])

print("\n--- 2.3.2 Outliers en UnitPrice (método IQR) ---")
Q1_price = df['UnitPrice'].quantile(0.25)
Q3_price = df['UnitPrice'].quantile(0.75)
IQR_price = Q3_price - Q1_price
limite_inf_price = Q1_price - 1.5 * IQR_price
limite_sup_price = Q3_price + 1.5 * IQR_price
outliers_price = df[(df['UnitPrice'] < limite_inf_price) | (df['UnitPrice'] > limite_sup_price)]
print(f"  Q1: {Q1_price} | Q3: {Q3_price} | IQR: {IQR_price}")
print(f"  Limite inferior: {limite_inf_price} | Limite superior: {limite_sup_price}")
print(f"  Total outliers en UnitPrice: {len(outliers_price)} ({len(outliers_price) / len(df) * 100:.2f}%)")
print(f"  Valores extremos (top 5 mayores):")
print(df.nlargest(5, 'UnitPrice')[['InvoiceNo', 'StockCode', 'Description', 'Quantity', 'UnitPrice']])

print("\n--- 2.3.3 StockCodes menos frecuentes (posibles outliers categóricos) ---")
stock_freq = df['StockCode'].value_counts()
stock_raros = stock_freq[stock_freq <= 3]
print(f"  StockCodes con 3 o menos apariciones: {len(stock_raros)}")
print(f"  Ejemplos:")
print(stock_raros.head(10))

print("\n--- 2.3.4 Descriptions menos frecuentes (posibles outliers categóricos) ---")
desc_freq = df['Description'].value_counts()
desc_raras = desc_freq[desc_freq <= 3]
print(f"  Descriptions con 3 o menos apariciones: {len(desc_raras)}")
print(f"  Ejemplos:")
print(desc_raras.head(10))

print("\n--- 2.3.5 Distribución de Quantity por percentiles ---")
percentiles = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1.0]
print(df['Quantity'].quantile(percentiles))

print("\n--- 2.3.6 Distribución de UnitPrice por percentiles ---")
print(df['UnitPrice'].quantile(percentiles))

