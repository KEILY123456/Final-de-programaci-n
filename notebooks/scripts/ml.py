import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder


def entrenar_modelo(data):
    # Convertirmos las variables categóricas en variables numéricas
    encoder = OneHotEncoder(sparse_output=False)
    X_encoded = encoder.fit_transform(data[['GrupoEdad', 'SexDSC']])

    y = data['HasAnnotations']

    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Precisión del modelo: {accuracy}')

    return model


if __name__ == "__main__":
    ruta_datos = 'data_transformada.csv'
    datos = pd.read_csv(ruta_datos)
    modelo = entrenar_modelo(datos)
