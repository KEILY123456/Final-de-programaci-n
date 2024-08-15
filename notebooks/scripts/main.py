from scripts.etl import load_data, transform_data, save_data
from scripts.ml import split_data, train_model, evaluate_model

# Paso 1: Cargar datos
df = load_data('data/raw_data.csv')

# Paso 2: Transformar datos
df_transformed = transform_data(df)
save_data(df_transformed, 'data/transformed_data.csv')

# Paso 3: Dividir datos
X_train, X_test, y_train, y_test = split_data(df_transformed, 'target_column')

# Paso 4: Entrenar modelo
model = train_model(X_train, y_train)

# Paso 5: Evaluar modelo
accuracy = evaluate_model(model, X_test, y_test)
print(f'Accuracy del modelo: {accuracy:.2f}')
