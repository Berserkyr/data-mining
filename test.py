import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report


def fetch_btc_data(days=1):
    """
    Récupère les données historiques du Bitcoin et resample par minute.
    """
    url = f'https://api.coingecko.com/api/v3/coins/bitcoin/market_chart'
    params = {'vs_currency': 'usd', 'days': days}
    response = requests.get(url, params=params)

    if response.status_code != 200:
        raise ValueError(f"Erreur HTTP : {response.status_code}, message : {response.text}")

    data = response.json()
    if 'prices' not in data:
        raise KeyError("La clé 'prices' est absente. Vérifiez l'API.")

    prices = data['prices']
    df = pd.DataFrame(prices, columns=['timestamp', 'price'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df.resample('min').mean()
    df.reset_index(inplace=True)

    return df


def fill_missing_with_regression(df, column='price', degree=2):
    """
    Remplit les valeurs manquantes dans une colonne avec une régression polynomiale.
    """
    valid_idx = df[column].notnull()
    missing_idx = df[column].isnull()

    # Convertir le temps en une variable numérique
    df['numeric_time'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()

    # Points d'entraînement
    X_train = df.loc[valid_idx, 'numeric_time'].values.reshape(-1, 1)
    y_train = df.loc[valid_idx, column].values

    # Points à prédire
    X_predict = df.loc[missing_idx, 'numeric_time'].values.reshape(-1, 1)

    # Régression polynomiale
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_predict_poly = poly.transform(X_predict)

    # Entraîner et prédire
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    y_predict = model.predict(X_predict_poly)

    # Remplir les valeurs manquantes
    df.loc[missing_idx, column] = y_predict
    df.drop(columns=['numeric_time'], inplace=True)

    return df


def prepare_data(df, n=10):
    """
    Prépare les données pour le modèle de classification.
    """
    df['return'] = df['price'].pct_change(fill_method=None)
    df['target'] = (df['return'] > 0).astype(int)

    for i in range(1, n + 1):
        df[f'return_lag_{i}'] = df['return'].shift(i)

    # Ajout de moyennes mobiles comme features
    df['sma_5'] = df['price'].rolling(window=5).mean()
    df['sma_10'] = df['price'].rolling(window=10).mean()

    df = df.dropna()
    return df


def train_model(X, y):
    """
    Entraîne un modèle Random Forest pour prédire les mouvements de prix.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42, class_weight='balanced'), param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Précision du modèle : {accuracy:.2%}")
    print("Rapport de classification :")
    print(classification_report(y_test, y_pred))

    return best_model, scaler


def predict_next_movement(data, model, scaler, n):
    """
    Prédit la direction du prix pour la prochaine période.
    """
    recent_data = data.iloc[-n:].filter(regex='return_lag|sma').values
    recent_data = scaler.transform(recent_data)

    prediction = model.predict(recent_data)[0]
    return "Hausse" if prediction == 1 else "Baisse"


# Script principal
if __name__ == "__main__":
    try:
        df = fetch_btc_data(days=1)
        print("Données chargées avec succès.")
    except Exception as e:
        print(f"Erreur lors de la récupération des données : {e}")
        exit()

    # Remplir les valeurs manquantes
    df = fill_missing_with_regression(df, column='price', degree=2)

    # Préparer les données
    n_features = 10
    df = prepare_data(df, n=n_features)

    # Définir les features et la cible
    feature_cols = [f'return_lag_{i}' for i in range(1, n_features + 1)] + ['sma_5', 'sma_10']
    X = df[feature_cols]
    y = df['target']

    # Entraîner le modèle
    model, scaler = train_model(X, y)

    # Faire une prédiction sur la prochaine période
    prediction = predict_next_movement(df, model, scaler, n_features)
    print("Prédiction pour la prochaine période :", prediction)
