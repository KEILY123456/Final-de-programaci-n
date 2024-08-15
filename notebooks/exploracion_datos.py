import pandas as pd

# Cargamos los datos del csv
file_path = 'data_prueba.csv'
data = pd.read_csv(file_path)

# Exploración inicial
print("Información: ", data.info())
print("")
print("Descripción: ", data.describe())
print("")
print("Cabecera: ", data.head())