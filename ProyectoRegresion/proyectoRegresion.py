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
