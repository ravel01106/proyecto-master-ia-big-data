# Comandos del proyecto

```bash
# Ejecutar el script principal
python proyectoRegresion.py

# Ejecutar y filtrar por sección (ej: sección 2.2, 2.3...)
python proyectoRegresion.py 2>&1 | Select-String "2\.2" -Context 0,60
python proyectoRegresion.py 2>&1 | Select-String "2\.3" -Context 0,70

# Verificar instalación de una librería
pip show pandas
pip show seaborn
pip show matplotlib

# Instalar librerías
pip install seaborn matplotlib
```
