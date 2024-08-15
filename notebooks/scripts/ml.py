from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def split_data(df, target_column):
    """Divide los datos en conjuntos de entrenamiento y prueba."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    """Entrena un modelo de Random Forest."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evalúa el modelo en el conjunto de prueba."""
    predictions = model.predict(X_test)
    return accuracy_score(y_test, predictions)
