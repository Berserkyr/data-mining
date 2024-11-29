import pandas as pd
import os

INPUT_DIR = 'data'
OUTPUT_DIR = 'cleaned_data'

# Assurez-vous que le répertoire de sortie existe
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Fonction de nettoyage des données avec journalisation
def clean_data(file_path):
    df = pd.read_csv(file_path)
    original_row_count = len(df)
    print(f"Original row count for {file_path}: {original_row_count}")
    
    # Élimination des espaces vides
    df.dropna(inplace=True)
    after_dropna_count = len(df)
    print(f"Rows after dropna for {file_path}: {after_dropna_count} (Removed {original_row_count - after_dropna_count} rows)")
    
    # Normalisation des types de données
    df = df.apply(pd.to_numeric, errors='coerce')
    after_normalization_count = len(df.dropna())
    print(f"Rows after normalization for {file_path}: {after_normalization_count} (Removed {after_dropna_count - after_normalization_count} rows)")
    
    # Suppression des lignes avec des types incorrects après normalisation
    df.dropna(inplace=True)
    cleaned_row_count = len(df)
    print(f"Cleaned row count for {file_path}: {cleaned_row_count} (Removed {after_normalization_count - cleaned_row_count} rows)")
    
    return df

# Nettoyage des données Fred
fred_file = os.path.join(INPUT_DIR, 'fred_sp500.csv')
cleaned_fred_data = clean_data(fred_file)
cleaned_fred_data.to_csv(os.path.join(OUTPUT_DIR, 'cleaned_fred_sp500.csv'), index=False)

# Nettoyage des données Yahoo Finance
yahoo_file = os.path.join(INPUT_DIR, 'yahoo_aapl.csv')
cleaned_yahoo_data = clean_data(yahoo_file)
cleaned_yahoo_data.to_csv(os.path.join(OUTPUT_DIR, 'cleaned_yahoo_aapl.csv'), index=False)

print("Données nettoyées et enregistrées avec succès.")
