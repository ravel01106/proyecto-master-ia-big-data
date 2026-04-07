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

