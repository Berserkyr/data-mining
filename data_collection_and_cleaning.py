import pandas as pd
import yfinance as yf
from fredapi import Fred
import os
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


# Configuration
FRED_API_KEY = '2c99ab88dfc264d83902305f71ad43d8'  # Remplacez par votre clé API Fred
OUTPUT_DIR = 'data'

# Assurez-vous que le répertoire de sortie existe
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Fonction pour collecter les données de Fred
def fetch_fred_data(series_id, start_date, end_date):
    fred = Fred(api_key=FRED_API_KEY)
    data = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
    df = pd.DataFrame(data, columns=['value'])
    df.index.name = 'date'
    return df

# Fonction pour collecter les données de Yahoo Finance
def fetch_yahoo_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Dates de début et de fin pour la collecte
start_date = '2020-01-01'
end_date = datetime.today().strftime('%Y-%m-%d')

# Collecte des données de Fred (exemple avec 'SP500')
fred_data = fetch_fred_data('SP500', start_date, end_date)
print("Collected Fred Data (first 5 rows):")
print(fred_data.head())
fred_data.to_csv(os.path.join(OUTPUT_DIR, 'fred_sp500.csv'))

# Collecte des données de Yahoo Finance (exemple avec 'AAPL')
yahoo_data = fetch_yahoo_data('AAPL', start_date, end_date)
print("Collected Yahoo Finance Data (first 5 rows):")
print(yahoo_data.head())
yahoo_data.to_csv(os.path.join(OUTPUT_DIR, 'yahoo_aapl.csv'))

# Nettoyage des données
def clean_data(df):
    print("Original Data (first 5 rows):")
    print(df.head())

    

    # Normalisation des types de données
    df = df.apply(pd.to_numeric, errors='coerce')
    df.dropna(inplace=True)
    print("After normalization (first 5 rows):")
    print(df.head())

    return df
# Fonction pour transformer les données
def transform_data(df):
    # Calculer les variations journalières en pourcentage
    df['daily_return'] = df['value'].pct_change()
    return df
# Nettoyage des données Fred
cleaned_fred_data = clean_data(fred_data)
cleaned_fred_data.to_csv(os.path.join(OUTPUT_DIR, 'cleaned_fred_sp500.csv'))

# Nettoyage des données Yahoo Finance
cleaned_yahoo_data = clean_data(yahoo_data)
cleaned_yahoo_data.to_csv(os.path.join(OUTPUT_DIR, 'cleaned_yahoo_aapl.csv'))

print("Données nettoyées et enregistrées avec succès.")

