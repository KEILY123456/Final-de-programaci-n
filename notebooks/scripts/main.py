from etl import cargar_datos, transformar_datos, guardar_datos
from ml import entrenar_modelo


def main():
    # Ruta de los datos
    ruta_entrada = 'data_prueba.csv'
    ruta_salida = 'data_transformada.csv'

    # Proceso ETL
    datos = cargar_datos(ruta_entrada)
    datos_transformados = transformar_datos(datos)
    guardar_datos(datos_transformados, ruta_salida)

    # Entrenamiento del modelo
    modelo = entrenar_modelo(datos_transformados)


if __name__ == "__main__":
    main()
