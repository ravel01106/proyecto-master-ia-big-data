import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

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

# 2.4 GRÁFICOS AUXILIARES

print("\n\n=== 2.4 CREACIÓN DE GRÁFICOS AUXILIARES ===")

sns.set_theme(style='whitegrid', palette='muted')
RUTA_GRAFICOS = 'graficos/'
import os
os.makedirs(RUTA_GRAFICOS, exist_ok=True)

print("\n  Generando gráfico 2.4.1 - Valores faltantes por columna...")
nulos = df.isnull().sum()
nulos = nulos[nulos > 0]

fig, ax = plt.subplots(figsize=(7, 4))
sns.barplot(x=nulos.index, y=nulos.values, ax=ax)
ax.set_title('Valores faltantes por columna', fontsize=14)
ax.set_xlabel('Columna')
ax.set_ylabel('Nº de valores nulos')
for bar in ax.patches:
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 500,
            f'{int(bar.get_height()):,}',
            ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig(f'{RUTA_GRAFICOS}2.4.1_valores_faltantes.png', dpi=150)
plt.show()
plt.close()
print("  Guardado: 2.4.1_valores_faltantes.png")

print("\n  Generando gráfico 2.4.2 - Distribución de Quantity (rango normal)...")
qty_filtrado = df[(df['Quantity'] > 0) & (df['Quantity'] <= 100)]

fig, ax = plt.subplots(figsize=(9, 4))
sns.histplot(qty_filtrado['Quantity'], bins=50, ax=ax, kde=True)
ax.set_title('Distribución de Quantity (0–100 uds.)', fontsize=14)
ax.set_xlabel('Quantity')
ax.set_ylabel('Frecuencia')
plt.tight_layout()
plt.savefig(f'{RUTA_GRAFICOS}2.4.2_distribucion_quantity.png', dpi=150)
plt.show()
plt.close()
print("  Guardado: 2.4.2_distribucion_quantity.png")

print("\n  Generando gráfico 2.4.3 - Distribución de UnitPrice (rango normal)...")
price_filtrado = df[(df['UnitPrice'] > 0) & (df['UnitPrice'] <= 20)]

fig, ax = plt.subplots(figsize=(9, 4))
sns.histplot(price_filtrado['UnitPrice'], bins=50, ax=ax, kde=True, color='coral')
ax.set_title('Distribución de UnitPrice (0–20 €)', fontsize=14)
ax.set_xlabel('UnitPrice (€)')
ax.set_ylabel('Frecuencia')
plt.tight_layout()
plt.savefig(f'{RUTA_GRAFICOS}2.4.3_distribucion_unitprice.png', dpi=150)
plt.show()
plt.close()
print("  Guardado: 2.4.3_distribucion_unitprice.png")

print("\n  Generando gráfico 2.4.4 - Top 10 países por transacciones...")
top_paises = df['Country'].value_counts().head(10)

fig, ax = plt.subplots(figsize=(9, 5))
sns.barplot(x=top_paises.values, y=top_paises.index, ax=ax, orient='h')
ax.set_title('Top 10 países por número de transacciones', fontsize=14)
ax.set_xlabel('Nº de transacciones')
ax.set_ylabel('País')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
plt.tight_layout()
plt.savefig(f'{RUTA_GRAFICOS}2.4.4_top10_paises.png', dpi=150)
plt.show()
plt.close()
print("  Guardado: 2.4.4_top10_paises.png")

print("\n  Generando gráfico 2.4.5 - Proporción de transacciones anómalas...")
total = len(df)
n_qty_neg    = (df['Quantity'] <= 0).sum()
n_price_neg  = (df['UnitPrice'] <= 0).sum()
n_sin_client = df['CustomerID'].isnull().sum()
n_duplicados = df.duplicated().sum()
n_normales   = total - n_qty_neg - n_price_neg - n_sin_client - n_duplicados

labels  = ['Normales', 'Quantity ≤ 0', 'UnitPrice ≤ 0', 'Sin CustomerID', 'Duplicados']
valores = [n_normales, n_qty_neg, n_price_neg, n_sin_client, n_duplicados]
colores = ['#4CAF50', '#F44336', '#FF9800', '#2196F3', '#9C27B0']

fig, ax = plt.subplots(figsize=(8, 6))
ax.pie(valores, labels=labels, colors=colores, autopct='%1.1f%%', startangle=140)
ax.set_title('Proporción de transacciones por tipo de anomalía', fontsize=14)
plt.tight_layout()
plt.savefig(f'{RUTA_GRAFICOS}2.4.5_proporcion_anomalias.png', dpi=150)
plt.show()
plt.close()
print("  Guardado: 2.4.5_proporcion_anomalias.png")

print("\n  Generando gráfico 2.4.6 - Boxplots Quantity y UnitPrice...")
qty_box   = df[(df['Quantity'] > 0) & (df['Quantity'] <= 100)]['Quantity']
price_box = df[(df['UnitPrice'] > 0) & (df['UnitPrice'] <= 20)]['UnitPrice']

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
sns.boxplot(y=qty_box, ax=axes[0], color='steelblue')
axes[0].set_title('Boxplot Quantity (0–100)', fontsize=13)
axes[0].set_ylabel('Quantity')

sns.boxplot(y=price_box, ax=axes[1], color='coral')
axes[1].set_title('Boxplot UnitPrice (0–20 €)', fontsize=13)
axes[1].set_ylabel('UnitPrice (€)')

plt.tight_layout()
plt.savefig(f'{RUTA_GRAFICOS}2.4.6_boxplots.png', dpi=150)
plt.show()
plt.close()
print("  Guardado: 2.4.6_boxplots.png")

print("\n  Todos los gráficos guardados en la carpeta 'graficos/'")

# 2.5 y 2.6 ANÁLISIS TEMPORAL y Análisis exploratorio de los datos y su calidad

print("\n\n=== 2.5 ANÁLISIS TEMPORAL ===")

print("\n--- 2.5.1 Conversión de InvoiceDate a datetime ---")
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='mixed')
print(f"  Tipo resultante: {df['InvoiceDate'].dtype}")
print(f"  Fecha mínima: {df['InvoiceDate'].min()}")
print(f"  Fecha máxima: {df['InvoiceDate'].max()}")
print(f"  Rango total: {(df['InvoiceDate'].max() - df['InvoiceDate'].min()).days} días")

df['Fecha'] = df['InvoiceDate'].dt.normalize()

print("\n--- 2.5.3 Transacciones por día ---")
trans_por_dia = df.groupby('Fecha').size().rename('NumTransacciones')
print(f"  Días con datos: {len(trans_por_dia)}")
print(f"  Media transacciones/día: {trans_por_dia.mean():.1f}")
print(f"  Máximo transacciones en un día: {trans_por_dia.max()} ({trans_por_dia.idxmax().date()})")
print(f"  Mínimo transacciones en un día: {trans_por_dia.min()} ({trans_por_dia.idxmin().date()})")

print("\n--- 2.5.4 Días sin datos en el rango ---")
rango_completo = pd.date_range(start=df['Fecha'].min(), end=df['Fecha'].max(), freq='D')
dias_sin_datos = rango_completo.difference(trans_por_dia.index)
print(f"  Total días en el rango: {len(rango_completo)}")
print(f"  Días con datos: {len(trans_por_dia)}")
print(f"  Días sin datos: {len(dias_sin_datos)}")
print(f"  Listado de días sin datos:")
print(dias_sin_datos.strftime('%Y-%m-%d').tolist())

print("\n  Generando gráfico 2.5.5 - Evolución de transacciones diarias...")
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(trans_por_dia.index, trans_por_dia.values, linewidth=1, color='steelblue')
ax.set_title('Evolución de transacciones diarias (dic 2010 – dic 2011)', fontsize=14)
ax.set_xlabel('Fecha')
ax.set_ylabel('Nº de transacciones')
ax.axvline(pd.Timestamp('2011-11-25'), color='orange', linestyle='--', linewidth=1.2, label='Black Friday 2011')
ax.axvline(pd.Timestamp('2011-12-01'), color='red',    linestyle='--', linewidth=1.2, label='Diciembre (test set)')
ax.legend()
plt.tight_layout()
plt.savefig(f'{RUTA_GRAFICOS}2.5.5_transacciones_diarias.png', dpi=150)
plt.show()
plt.close()
print("  Guardado: 2.5.5_transacciones_diarias.png")

print("\n  Generando gráfico 2.5.6 - Transacciones por mes...")
df['Mes'] = df['InvoiceDate'].dt.to_period('M')
trans_por_mes = df.groupby('Mes').size().rename('NumTransacciones')

fig, ax = plt.subplots(figsize=(12, 5))
trans_por_mes.plot(kind='bar', ax=ax, color='steelblue', edgecolor='white')
ax.set_title('Transacciones por mes', fontsize=14)
ax.set_xlabel('Mes')
ax.set_ylabel('Nº de transacciones')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig(f'{RUTA_GRAFICOS}2.5.6_transacciones_por_mes.png', dpi=150)
plt.show()
plt.close()
print("  Guardado: 2.5.6_transacciones_por_mes.png")

print("\n  Generando gráfico 2.5.7 - Transacciones por día de la semana...")
dias_semana = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
df['DiaSemana'] = df['InvoiceDate'].dt.day_name()
trans_por_dia_semana = df.groupby('DiaSemana').size().reindex(dias_semana).rename('NumTransacciones')

fig, ax = plt.subplots(figsize=(9, 5))
sns.barplot(x=trans_por_dia_semana.index, y=trans_por_dia_semana.values, ax=ax)
ax.set_title('Transacciones por día de la semana', fontsize=14)
ax.set_xlabel('Día')
ax.set_ylabel('Nº de transacciones')
plt.tight_layout()
plt.savefig(f'{RUTA_GRAFICOS}2.5.7_transacciones_dia_semana.png', dpi=150)
plt.show()
plt.close()
print("  Guardado: 2.5.7_transacciones_dia_semana.png")

# 2.6 ANÁLISIS DE CANCELACIONES (prefijo "C" en InvoiceNo)

print("\n\n=== 2.6 ANÁLISIS DE CANCELACIONES ===")

print("\n--- 2.6.1 Facturas con prefijo 'C' (cancelaciones) ---")
df['EsCancelacion'] = df['InvoiceNo'].str.startswith('C')
n_cancelaciones = df['EsCancelacion'].sum()
print(f"  Total filas con prefijo 'C': {n_cancelaciones}")
print(f"  % sobre total: {n_cancelaciones / len(df) * 100:.2f}%")
print(f"  Facturas únicas canceladas: {df[df['EsCancelacion']]['InvoiceNo'].nunique()}")

print("\n--- 2.6.2 Cruce entre prefijo 'C' y Quantity < 0 ---")
con_C_qty_neg    = df[ df['EsCancelacion'] &  (df['Quantity'] < 0)]
con_C_qty_pos    = df[ df['EsCancelacion'] & (df['Quantity'] >= 0)]
sin_C_qty_neg    = df[~df['EsCancelacion'] &  (df['Quantity'] < 0)]
sin_C_qty_pos    = df[~df['EsCancelacion'] & (df['Quantity'] >= 0)]

print(f"  Prefijo 'C' + Quantity < 0  (cancelaciones normales):       {len(con_C_qty_neg):>7,}")
print(f"  Prefijo 'C' + Quantity >= 0 (cancelaciones con qty positiva):{len(con_C_qty_pos):>7,}")
print(f"  Sin 'C'    + Quantity < 0  (negativos huérfanos):            {len(sin_C_qty_neg):>7,}")
print(f"  Sin 'C'    + Quantity >= 0 (transacciones normales):         {len(sin_C_qty_pos):>7,}")

print("\n--- 2.6.3 Detalle de negativos huérfanos (sin prefijo 'C') ---")
print(f"  Total: {len(sin_C_qty_neg)}")
if len(sin_C_qty_neg) > 0:
    print(sin_C_qty_neg[['InvoiceNo', 'StockCode', 'Description', 'Quantity', 'UnitPrice', 'CustomerID']].head(10))
    print(f"\n  StockCodes más frecuentes en huérfanos:")
    print(sin_C_qty_neg['StockCode'].value_counts().head(10))

print("\n--- 2.6.4 Cancelaciones con Quantity >= 0 (anomalía) ---")
print(f"  Total: {len(con_C_qty_pos)}")
if len(con_C_qty_pos) > 0:
    print(con_C_qty_pos[['InvoiceNo', 'StockCode', 'Description', 'Quantity', 'UnitPrice']].head(10))

print("\n--- 2.6.5 Cancelaciones por mes ---")
cancelaciones_mes = df[df['EsCancelacion']].groupby('Mes').size().rename('Cancelaciones')
total_mes         = df.groupby('Mes').size().rename('Total')
ratio_cancelacion = (cancelaciones_mes / total_mes * 100).round(2).rename('% Cancelaciones')
print(pd.concat([total_mes, cancelaciones_mes, ratio_cancelacion], axis=1))

print("\n  Generando gráfico 2.6.6 - Proporción cancelaciones vs normales...")
# Excluimos categorías con 0 para evitar solapamiento de etiquetas
datos_grafico = {
    'Normales':                   len(sin_C_qty_pos),
    'Cancelaciones (C + Qty<0)':  len(con_C_qty_neg),
    'Huérfanos (sin C + Qty<0)':  len(sin_C_qty_neg),
}
labels  = list(datos_grafico.keys())
valores = list(datos_grafico.values())
colores = ['#4CAF50', '#F44336', '#9C27B0']

fig, ax = plt.subplots(figsize=(9, 7))
wedges, texts, autotexts = ax.pie(
    valores,
    colors=colores,
    autopct=lambda p: f'{p:.1f}%\n({int(p * sum(valores) / 100):,})',
    startangle=140,
    pctdistance=0.75,
    wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
)
for autotext in autotexts:
    autotext.set_fontsize(10)
ax.legend(wedges, labels, loc='lower center', bbox_to_anchor=(0.5, -0.08),
          ncol=1, fontsize=10, frameon=True)
ax.set_title('Tipos de transacciones: normales vs cancelaciones', fontsize=13, pad=20)
plt.tight_layout()
plt.savefig(f'{RUTA_GRAFICOS}2.6.6_cancelaciones_proporcion.png', dpi=150)
plt.show()
plt.close()
print("  Guardado: 2.6.6_cancelaciones_proporcion.png")

print("\n  Generando gráfico 2.6.7 - Tasa de cancelación por mes...")
fig, ax = plt.subplots(figsize=(12, 5))
ratio_cancelacion.plot(kind='bar', ax=ax, color='salmon', edgecolor='white')
ax.set_title('Tasa de cancelación mensual (%)', fontsize=14)
ax.set_xlabel('Mes')
ax.set_ylabel('% cancelaciones sobre total')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig(f'{RUTA_GRAFICOS}2.6.7_tasa_cancelacion_mensual.png', dpi=150)
plt.show()
plt.close()
print("  Guardado: 2.6.7_tasa_cancelacion_mensual.png")

# 2.7 VENTAS DIARIAS BRUTAS (variable objetivo preliminar)

print("\n\n=== 2.7 VENTAS DIARIAS BRUTAS (TotalPrice) ===")

print("\n--- 2.7.1 Cálculo de TotalPrice = Quantity × UnitPrice ---")
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
print(f"  Estadísticas de TotalPrice por fila:")
print(df['TotalPrice'].describe())

print("\n--- 2.7.2 Agregación de ventas por día (bruto, sin limpiar) ---")
ventas_diarias = df.groupby('Fecha')['TotalPrice'].sum().rename('VentasDiarias')
print(f"  Días con datos: {len(ventas_diarias)}")
print(f"  Venta media diaria:  £{ventas_diarias.mean():>12,.2f}")
print(f"  Venta mediana diaria: £{ventas_diarias.median():>11,.2f}")
print(f"  Venta máxima diaria:  £{ventas_diarias.max():>11,.2f} ({ventas_diarias.idxmax().date()})")
print(f"  Venta mínima diaria:  £{ventas_diarias.min():>11,.2f} ({ventas_diarias.idxmin().date()})")
print(f"  Días con ventas negativas: {(ventas_diarias < 0).sum()}")

print("\n--- 2.7.3 Ventas totales por mes ---")
ventas_mes = df.groupby('Mes')['TotalPrice'].sum().rename('VentasMes')
print(ventas_mes.apply(lambda x: f'£{x:,.2f}'))

print("\n--- 2.7.4 Distribución de ventas diarias por percentiles ---")
percentiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
print(ventas_diarias.quantile(percentiles).apply(lambda x: f'£{x:,.2f}'))

print("\n  Generando gráfico 2.7.5 - Evolución de ventas diarias brutas...")
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(ventas_diarias.index, ventas_diarias.values, linewidth=1, color='steelblue')
ax.axhline(ventas_diarias.mean(), color='orange', linestyle='--', linewidth=1.2, label=f'Media: £{ventas_diarias.mean():,.0f}')
ax.axvline(pd.Timestamp('2011-11-09'), color='green', linestyle='--', linewidth=1.2, label='Inicio test set (9 nov)')
ax.set_title('Evolución de ventas diarias brutas — variable objetivo (sin limpiar)', fontsize=13)
ax.set_xlabel('Fecha')
ax.set_ylabel('Ventas (£)')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'£{x:,.0f}'))
ax.legend()
plt.tight_layout()
plt.savefig(f'{RUTA_GRAFICOS}2.7.5_ventas_diarias_brutas.png', dpi=150)
plt.show()
plt.close()
print("  Guardado: 2.7.5_ventas_diarias_brutas.png")

print("\n  Generando gráfico 2.7.6 - Ventas totales por mes...")
fig, ax = plt.subplots(figsize=(12, 5))
ventas_mes.plot(kind='bar', ax=ax, color='steelblue', edgecolor='white')
ax.set_title('Ventas totales brutas por mes (£)', fontsize=14)
ax.set_xlabel('Mes')
ax.set_ylabel('Ventas (£)')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'£{x/1e6:.1f}M'))
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig(f'{RUTA_GRAFICOS}2.7.6_ventas_mensuales_brutas.png', dpi=150)
plt.show()
plt.close()
print("  Guardado: 2.7.6_ventas_mensuales_brutas.png")

print("\n  Generando gráfico 2.7.7 - Distribución de ventas diarias...")
ventas_positivas = ventas_diarias[ventas_diarias > 0]
fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(ventas_positivas, bins=40, ax=ax, kde=True, color='steelblue')
ax.axvline(ventas_positivas.mean(),   color='orange', linestyle='--', linewidth=1.5, label=f'Media: £{ventas_positivas.mean():,.0f}')
ax.axvline(ventas_positivas.median(), color='green',  linestyle='--', linewidth=1.5, label=f'Mediana: £{ventas_positivas.median():,.0f}')
ax.set_title('Distribución de ventas diarias brutas (días con ventas > 0)', fontsize=13)
ax.set_xlabel('Ventas (£)')
ax.set_ylabel('Frecuencia')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'£{x:,.0f}'))
ax.legend()
plt.tight_layout()
plt.savefig(f'{RUTA_GRAFICOS}2.7.7_distribucion_ventas_diarias.png', dpi=150)
plt.show()
plt.close()
print("  Guardado: 2.7.7_distribucion_ventas_diarias.png")

# 2.8 TOP CLIENTES Y TOP PRODUCTOS

print("\n\n=== 2.8 TOP CLIENTES Y TOP PRODUCTOS ===")

print("\n--- 2.8.1 Top 10 clientes por volumen de ventas (TotalPrice) ---")
ventas_cliente = (
    df[df['CustomerID'].notna()]
    .groupby('CustomerID')['TotalPrice']
    .sum()
    .sort_values(ascending=False)
)
top10_clientes = ventas_cliente.head(10)
total_ventas   = df['TotalPrice'].sum()
print(f"  Ventas totales brutas: £{total_ventas:,.2f}")
print(f"\n  {'CustomerID':<15} {'Ventas (£)':>15} {'% sobre total':>15}")
print(f"  {'-'*45}")
for cid, ventas in top10_clientes.items():
    print(f"  {int(cid):<15} £{ventas:>13,.2f} {ventas / total_ventas * 100:>14.2f}%")
pct_top10 = top10_clientes.sum() / total_ventas * 100
print(f"\n  Top 10 clientes concentran el {pct_top10:.1f}% de las ventas totales")

print("\n--- 2.8.2 Top 10 productos por volumen de ventas (TotalPrice) ---")
ventas_producto = (
    df.groupby('StockCode')['TotalPrice']
    .sum()
    .sort_values(ascending=False)
)
top10_productos = ventas_producto.head(10)
print(f"\n  {'StockCode':<12} {'Ventas (£)':>15} {'% sobre total':>15} {'Descripción'}")
print(f"  {'-'*75}")
for sc, ventas in top10_productos.items():
    desc = df[df['StockCode'] == sc]['Description'].dropna().mode()
    desc = desc.iloc[0] if len(desc) > 0 else 'N/A'
    print(f"  {sc:<12} £{ventas:>13,.2f} {ventas / total_ventas * 100:>14.2f}%  {desc[:35]}")
pct_top10_prod = top10_productos.sum() / total_ventas * 100
print(f"\n  Top 10 productos concentran el {pct_top10_prod:.1f}% de las ventas totales")

print("\n--- 2.8.3 Concentración acumulada de ventas (clientes) ---")
ventas_cliente_pos = ventas_cliente[ventas_cliente > 0].sort_values(ascending=False)
acumulado_pct      = ventas_cliente_pos.cumsum() / ventas_cliente_pos.sum() * 100
n_clientes_80      = (acumulado_pct <= 80).sum()
print(f"  Clientes con ventas positivas: {len(ventas_cliente_pos)}")
print(f"  Clientes que generan el 80% de las ventas: {n_clientes_80} ({n_clientes_80 / len(ventas_cliente_pos) * 100:.1f}%)")

print("\n  Generando gráfico 2.8.4 - Top 10 clientes por ventas...")
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x=[str(int(c)) for c in top10_clientes.index],
            y=top10_clientes.values, ax=ax, color='steelblue')
ax.set_title('Top 10 clientes por volumen de ventas brutas (£)', fontsize=13)
ax.set_xlabel('CustomerID')
ax.set_ylabel('Ventas (£)')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'£{x:,.0f}'))
ax.tick_params(axis='x', rotation=30)
plt.tight_layout()
plt.savefig(f'{RUTA_GRAFICOS}2.8.4_top10_clientes.png', dpi=150)
plt.show()
plt.close()
print("  Guardado: 2.8.4_top10_clientes.png")

print("\n  Generando gráfico 2.8.5 - Top 10 productos por ventas...")
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x=top10_productos.index, y=top10_productos.values, ax=ax, color='coral')
ax.set_title('Top 10 productos por volumen de ventas brutas (£)', fontsize=13)
ax.set_xlabel('StockCode')
ax.set_ylabel('Ventas (£)')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'£{x:,.0f}'))
ax.tick_params(axis='x', rotation=30)
plt.tight_layout()
plt.savefig(f'{RUTA_GRAFICOS}2.8.5_top10_productos.png', dpi=150)
plt.show()
plt.close()
print("  Guardado: 2.8.5_top10_productos.png")

print("\n  Generando gráfico 2.8.6 - Curva de Pareto de clientes...")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(1, len(acumulado_pct) + 1), acumulado_pct.values, color='steelblue', linewidth=1.5)
ax.axhline(80, color='orange', linestyle='--', linewidth=1.2, label='80% de ventas')
ax.axvline(n_clientes_80, color='red', linestyle='--', linewidth=1.2, label=f'{n_clientes_80} clientes')
ax.set_title('Curva de Pareto — concentración de ventas por cliente', fontsize=13)
ax.set_xlabel('Nº de clientes (ordenados por ventas desc.)')
ax.set_ylabel('% acumulado de ventas')
ax.legend()
plt.tight_layout()
plt.savefig(f'{RUTA_GRAFICOS}2.8.6_pareto_clientes.png', dpi=150)
plt.show()
plt.close()
print("  Guardado: 2.8.6_pareto_clientes.png")

# 3. LIMPIEZA DE DATOS

# Trabajamos sobre una copia para preservar el DataFrame original del EDA
df_clean = df.copy()
filas_iniciales = len(df_clean)
print(f"\n\n{'='*60}")
print(f"  INICIO LIMPIEZA — Filas iniciales: {filas_iniciales:,}")
print(f"{'='*60}")

# 3.1 ELIMINAR FILAS CON Description NULA

# Motivo: el 100% de estas filas cumplen simultáneamente:
#   - Description = NaN  → no sabemos qué producto es
#   - UnitPrice = 0      → no generan ningún ingreso (TotalPrice = 0)
#   - CustomerID = NaN   → no tienen cliente asociado
# No son recuperables y solo añadirían ruido al modelo.

print("\n--- 3.1 Eliminar filas con Description nula ---")

antes = len(df_clean)
df_clean = df_clean.dropna(subset=['Description']).reset_index(drop=True)
eliminadas = antes - len(df_clean)

print(f"  Filas antes:     {antes:,}")
print(f"  Filas eliminadas: {eliminadas:,}")
print(f"  Filas después:   {len(df_clean):,}")
print(f"  Verificación — Description nulos restantes: {df_clean['Description'].isnull().sum()}")

# 3.2 ELIMINAR DUPLICADOS EXACTOS
#
# Una fila duplicada exacta tiene TODOS sus campos idénticos: mismo InvoiceNo,
# StockCode, Description, Quantity, UnitPrice, InvoiceDate, CustomerID y Country.
# Esto es físicamente imposible en un sistema transaccional real: si el mismo cliente
# compra el mismo producto en el mismo instante, el sistema generaría un InvoiceNo
# distinto o un timestamp diferente. Su presencia indica errores de doble inserción
# en la BBDD, exports corruptos o fallos de ETL.

print("\n--- 3.2 Eliminar duplicados exactos ---")

antes = len(df_clean)
df_clean = df_clean.drop_duplicates(keep='first', ignore_index=True)
eliminadas = antes - len(df_clean)

print(f"  Filas antes:      {antes:,}")
print(f"  Filas eliminadas: {eliminadas:,}")
print(f"  Filas después:    {len(df_clean):,}")
print(f"  Verificación — duplicados restantes: {df_clean.duplicated().sum()}")

# 3.3 ELIMINAR NEGATIVOS HUÉRFANOS
#
# Son filas con Quantity < 0 pero SIN prefijo "C" en InvoiceNo.
# El análisis directo del CSV confirma que el 100% cumple simultáneamente:
#   - UnitPrice = 0.0  → TotalPrice = 0 siempre, sin impacto en ingresos
#   - CustomerID = NaN → ninguna tiene cliente asociado
#   - InvoiceNo sin "C" → el sistema nunca las registró como cancelación formal

print("\n--- 3.3 Eliminar negativos huérfanos (ajustes de almacén) ---")

antes = len(df_clean)
mask_huerfanos = (
    ~df_clean['InvoiceNo'].str.startswith('C', na=False) &
    (df_clean['Quantity'] < 0) &
    (df_clean['UnitPrice'] == 0.0)
)
df_clean = df_clean[~mask_huerfanos].reset_index(drop=True)
eliminadas = antes - len(df_clean)

print(f"  Filas antes:      {antes:,}")
print(f"  Filas eliminadas: {eliminadas:,}")
print(f"  Filas después:    {len(df_clean):,}")
print(f"  Verificación — negativos huérfanos restantes: {(~df_clean['InvoiceNo'].str.startswith('C', na=False) & (df_clean['Quantity'] < 0) & (df_clean['UnitPrice'] == 0.0)).sum()}")

# 3.4 ELIMINAR STOCKCODES NO ESTÁNDAR

print("\n--- 3.4 Eliminar StockCodes no estándar ---")

antes = len(df_clean)
mask_std = df_clean['StockCode'].str.match(r'^[0-9]{5}[A-Za-z]?$', na=False)
df_clean = df_clean[mask_std].reset_index(drop=True)
eliminadas = antes - len(df_clean)

print(f"  Filas antes:      {antes:,}")
print(f"  Filas eliminadas: {eliminadas:,}")
print(f"  Filas después:    {len(df_clean):,}")
n_no_std = (~df_clean['StockCode'].str.match(r'^[0-9]{5}[A-Za-z]?$', na=False)).sum()
print(f"  Verificación — StockCodes no estándar restantes: {n_no_std}")

# 3.5 CAPPING (WINSORIZACIÓN) DE OUTLIERS EN Quantity Y UnitPrice
#
# IMPORTANTE — TotalPrice se recalcula al final:
#   - df_clean heredó TotalPrice = Quantity × UnitPrice calculado ANTES del
#     capping (sección 2.7 sobre df original). Si no lo recalculamos quedaría
#     desincronizado con los nuevos valores de Quantity y UnitPrice.

print("\n--- 3.5 Capping de outliers (winsorización al percentil 99) ---")

cap_qty   = df_clean.loc[df_clean['Quantity']  > 0, 'Quantity'].quantile(0.99)
cap_price = df_clean.loc[df_clean['UnitPrice'] > 0, 'UnitPrice'].quantile(0.99)

print(f"  Umbral Quantity   (p99): {cap_qty:.1f} uds  →  clip [{-cap_qty:.1f}, {cap_qty:.1f}]")
print(f"  Umbral UnitPrice  (p99): £{cap_price:.2f}  →  clip [-, {cap_price:.2f}]")

n_qty_sup  = (df_clean['Quantity']  >  cap_qty).sum()
n_qty_inf  = (df_clean['Quantity']  < -cap_qty).sum()
n_price_sup = (df_clean['UnitPrice'] >  cap_price).sum()
print(f"  Filas Quantity  > +umbral (recortadas arriba): {n_qty_sup:,}")
print(f"  Filas Quantity  < -umbral (recortadas abajo):  {n_qty_inf:,}")
print(f"  Filas UnitPrice > umbral  (recortadas arriba): {n_price_sup:,}")

df_clean['Quantity']  = df_clean['Quantity'].clip(lower=-cap_qty, upper=cap_qty)
df_clean['UnitPrice'] = df_clean['UnitPrice'].clip(upper=cap_price)

# Recalcular TotalPrice con los valores ya capeados
df_clean['TotalPrice'] = df_clean['Quantity'] * df_clean['UnitPrice']

print(f"  Quantity  rango tras capping:  [{df_clean['Quantity'].min():.1f}, {df_clean['Quantity'].max():.1f}]")
print(f"  UnitPrice máxima tras capping: £{df_clean['UnitPrice'].max():.2f}")
print(f"  Filas totales (sin cambio):    {len(df_clean):,}")

print("\n  Estadísticas post-capping (validación de rangos):")
print(df_clean[['Quantity', 'UnitPrice', 'TotalPrice']].describe().round(2))

# 3.5b ELIMINAR FILAS CON UnitPrice = 0
#
# Tras los pasos 3.1–3.5, pueden quedar filas con UnitPrice = 0 que no son
# huérfanos (ya eliminados en 3.3) sino productos con precio cero: muestras,
# regalos o errores de registro. Todas generan TotalPrice = 0 con independencia
# de la Quantity → no aportan señal a la variable objetivo.
# Mantenerlas añadiría ruido al modelo sin representar ningún ingreso real.
# EXCEPCIÓN: las cancelaciones (prefijo C) pueden tener UnitPrice = 0 residual
# tras el capping; también se eliminan porque su TotalPrice = 0 igualmente.

print("\n--- 3.5b Eliminar filas con UnitPrice = 0 ---")

antes = len(df_clean)
df_clean = df_clean[df_clean['UnitPrice'] > 0].reset_index(drop=True)
eliminadas = antes - len(df_clean)

print(f"  Filas antes:      {antes:,}")
print(f"  Filas eliminadas: {eliminadas:,}")
print(f"  Filas después:    {len(df_clean):,}")
print(f"  Verificación — filas con UnitPrice = 0 restantes: {(df_clean['UnitPrice'] == 0).sum()}")

# 3.6 CONSERVAR FILAS CON CustomerID NULO
#
# Los ~135.080 registros sin CustomerID son ventas anónimas reales.
# Para predecir ventas diarias (variable objetivo = suma de TotalPrice por día)
# el CustomerID es irrelevante: la transacción genera ingresos con independencia
# de que el cliente esté identificado o no.
# Eliminarlos supondría perder ~25 % del dataset sin ningún beneficio para el modelo.

print("\n--- 3.6 CustomerID nulo — decisión: conservar ---")

n_sin_cliente = df_clean['CustomerID'].isnull().sum()
pct = n_sin_cliente / len(df_clean) * 100
print(f"  Filas con CustomerID nulo: {n_sin_cliente:,} ({pct:.2f}%)")
print(f"  Decisión: se conservan — son ventas anónimas válidas para la variable objetivo")
print(f"  Filas totales sin cambio:  {len(df_clean):,}")

# 3.7 TRATAR CANCELACIONES (prefijo "C" en InvoiceNo)
#
# Las cancelaciones tienen Quantity < 0 → TotalPrice < 0.
# NO se eliminan: al agregar por día con groupby('Fecha')['TotalPrice'].sum()
# se restan automáticamente de las ventas brutas de ese día, dando la venta NETA real.
# Ejemplo: si el lunes hay £10.000 en ventas y £500 en devoluciones,
#          el agregado diario da £9.500 sin ninguna acción adicional.
# Eliminarlas sobreestimaría las ventas diarias y el modelo aprendería
# una señal irreal (ventas brutas en lugar de ventas netas).

print("\n--- 3.7 Cancelaciones (prefijo C) — decisión: conservar con TotalPrice negativo ---")

mask_cancel = df_clean['InvoiceNo'].str.startswith('C', na=False)
n_cancel    = mask_cancel.sum()
tp_cancel   = df_clean.loc[mask_cancel, 'TotalPrice'].sum()
print(f"  Filas de cancelación en df_clean: {n_cancel:,}")
print(f"  TotalPrice acumulado cancelaciones: £{tp_cancel:,.2f}")
print(f"  Decisión: se conservan — el TotalPrice negativo reduce el agregado diario automáticamente")
print(f"  Filas totales sin cambio: {len(df_clean):,}")

# Verificación: ejemplo de un día con cancelaciones para confirmar el mecanismo
ventas_netas_dia = df_clean.groupby('Fecha')['TotalPrice'].sum()
dias_negativos   = (ventas_netas_dia < 0).sum()
print(f"  Días con venta neta negativa (devoluciones > ventas brutas): {dias_negativos}")

# 3.8 VERIFICAR INTEGRIDAD TEMPORAL TRAS LA LIMPIEZA
#
# Comprobamos que los pasos 3.1–3.7 no han eliminado accidentalmente días
# completos dentro del rango de entrenamiento (01/12/2010 → 08/11/2011)
# ni dentro del test set (09/11/2011 → 09/12/2011).
# Los días sin datos esperados son festivos y fines de semana — no días laborables.

print("\n--- 3.8 Integridad temporal tras la limpieza ---")

fechas_clean   = df_clean['Fecha'].drop_duplicates().sort_values()
fecha_min      = fechas_clean.min()
fecha_max      = fechas_clean.max()
rango_completo = pd.date_range(start=fecha_min, end=fecha_max, freq='D')
dias_sin_datos = rango_completo.difference(fechas_clean)

print(f"  Fecha mínima en df_clean: {fecha_min.date()}")
print(f"  Fecha máxima en df_clean: {fecha_max.date()}")
print(f"  Días totales en el rango: {len(rango_completo)}")
print(f"  Días con datos:           {len(fechas_clean)}")
print(f"  Días sin datos:           {len(dias_sin_datos)}")

# Verificar que el test set completo tiene datos
TEST_INICIO = pd.Timestamp('2011-11-09')
TEST_FIN    = pd.Timestamp('2011-12-09')
dias_test   = pd.date_range(start=TEST_INICIO, end=TEST_FIN, freq='D')
dias_test_sin_datos = dias_test.difference(fechas_clean)
print(f"\n  Test set ({TEST_INICIO.date()} → {TEST_FIN.date()}):")
print(f"    Días en el rango del test:      {len(dias_test)}")
print(f"    Días con datos en el test set:  {len(dias_test) - len(dias_test_sin_datos)}")
print(f"    Días SIN datos en el test set:  {len(dias_test_sin_datos)}")
if len(dias_test_sin_datos) > 0:
    print(f"    Días sin datos: {dias_test_sin_datos.strftime('%Y-%m-%d').tolist()}")
else:
    print(f"    ✓ Todos los días del test set tienen datos")

# Días sin datos esperados (festivos y fines de semana)
print(f"\n  Días sin datos en todo el rango (esperados: festivos/fines de semana):")
print(f"    {dias_sin_datos.strftime('%Y-%m-%d').tolist()}")

# 3.9 DATASET LIMPIO
#
# Comparativa paso a paso de filas eliminadas y guardado del CSV limpio.

print(f"\n\n{'='*60}")
print(f"  RESUMEN LIMPIEZA — COMPARATIVA POR PASO")
print(f"{'='*60}")

pasos = [
    ("Filas originales",                     filas_iniciales,           0),
    ("3.1 Eliminar Description nula",        540_455,  filas_iniciales - 540_455),
    ("3.2 Eliminar duplicados exactos",       535_187,  540_455 - 535_187),
    ("3.3 Eliminar negativos huérfanos",      533_851,  535_187 - 533_851),
    ("3.4 Eliminar StockCodes no estándar",   531_356,  533_851 - 531_356),
    ("3.5 Capping outliers (sin eliminar)",   531_356,  0),
    ("3.6 CustomerID nulo (conservar)",       531_356,  0),
    ("3.7 Cancelaciones (conservar)",         531_356,  0),
]

print(f"\n  {'Paso':<42} {'Filas':>8}  {'Eliminadas':>10}")
print(f"  {'-'*62}")
for nombre, filas, eliminadas in pasos:
    marca = "  " if eliminadas == 0 else "–>"
    print(f"  {marca} {nombre:<40} {filas:>8,}  {eliminadas:>10,}")

filas_finales   = len(df_clean)
total_eliminado = filas_iniciales - filas_finales
pct_eliminado   = total_eliminado / filas_iniciales * 100
print(f"\n  {'TOTAL ELIMINADAS':<42} {total_eliminado:>8,}  ({pct_eliminado:.2f}%)")
print(f"  {'FILAS FINALES EN df_clean':<42} {filas_finales:>8,}")

# Guardar CSV limpio
RUTA_CLEAN = 'contenidoCSV/data_clean.csv'
df_clean.to_csv(RUTA_CLEAN, index=False, encoding='utf-8')

print(f"\n--- 3.9 Dataset limpio guardado ---")
print(f"  Ruta:            {RUTA_CLEAN}")
print(f"  Filas:           {filas_finales:,}")
print(f"  Columnas:        {df_clean.shape[1]}")
print(f"  Columnas:        {list(df_clean.columns)}")
print(f"  Memoria (MB):    {df_clean.memory_usage(deep=True).sum() / 1024**2:.1f}")
