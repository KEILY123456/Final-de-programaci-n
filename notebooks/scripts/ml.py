import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
import numpy as np


def entrenar_modelo(data):
    # Convertimos las variables categóricas en variables numéricas
    encoder = OneHotEncoder(sparse_output=False)
    X_encoded = encoder.fit_transform(data[['GrupoEdad', 'SexDSC']])

    y = data['HasAnnotations']

    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Precisión del modelo: {accuracy}')

    # Importancia de características
    feature_importances = model.feature_importances_
    feature_names = encoder.get_feature_names_out(['GrupoEdad', 'SexDSC'])
    feature_df = pd.DataFrame({
        'Característica': feature_names,
        'Importancia': feature_importances
    })

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importancia', y='Característica', data=feature_df)
    plt.title('Importancia de las Características')
    plt.show()

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicho')
    plt.ylabel('Real')
    plt.title('Matriz de Confusión')
    plt.show()

    return model


def visualizar_distribucion(data):
    plt.figure(figsize=(12, 6))
    sns.countplot(x='GrupoEdad', hue='SexDSC', data=data)
    plt.title('Distribución de Grupo de Edad por Sexo')
    plt.show()


if __name__ == "__main__":
    ruta_datos = 'data_transformada.csv'
    datos = pd.read_csv(ruta_datos)

    visualizar_distribucion(datos)
    modelo = entrenar_modelo(datos)
