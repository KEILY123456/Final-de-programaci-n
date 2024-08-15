import pandas as pd


def cargar_datos(ruta):
    return pd.read_csv(ruta)


def transformar_datos(data):
    # Se convirtieron las edades a grupos etarios
    data['GrupoEdad'] = pd.cut(data['AgeAtVisit'], bins=[0, 18, 40, 60, 80, 100],
                               labels=['0-18', '19-40', '41-60', '61-80', '81-100'])
    return data


def guardar_datos(data, ruta):
    data.to_csv(ruta, index=False)


if __name__ == "__main__":
    ruta_entrada = 'data_prueba.csv'
    ruta_salida = 'data_transformada.csv'

    datos = cargar_datos(ruta_entrada)
    datos_transformados = transformar_datos(datos)
    guardar_datos(datos_transformados, ruta_salida)
