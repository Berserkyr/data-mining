import pandas as pd
import yfinance as yf
from fredapi import Fred
import os
from datetime import datetime

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
fred_data.to_csv(os.path.join(OUTPUT_DIR, 'fred_sp500.csv'))

# Collecte des données de Yahoo Finance (exemple avec 'AAPL')
yahoo_data = fetch_yahoo_data('AAPL', start_date, end_date)
yahoo_data.to_csv(os.path.join(OUTPUT_DIR, 'yahoo_aapl.csv'))

print("Données collectées et enregistrées avec succès.")
