import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px

# Chargement des données nettoyées
fred_data = pd.read_csv('data/cleaned_fred_sp500.csv')
yahoo_data = pd.read_csv('data/cleaned_yahoo_aapl.csv')

# Diagnostics pour vérifier les données
print("Fred Data (first 5 rows):")
print(fred_data.head())
print("\nYahoo Finance Data (first 5 rows):")
print(yahoo_data.head())

# Conversion des colonnes de date
fred_data['date'] = pd.to_datetime(fred_data['date'])
yahoo_data['Date'] = pd.to_datetime(yahoo_data['Date'])

# Diagnostics après conversion
print("Fred Data after date conversion (first 5 rows):")
print(fred_data.head())
print("\nYahoo Finance Data after date conversion (first 5 rows):")
print(yahoo_data.head())

# Application Dash
app = dash.Dash(__name__)

app.layout = html.Div(children=[
    html.H1(children='Visualisation des Données des Marchés Financiers'),

    dcc.Graph(
        id='fred-graph',
        figure=px.line(fred_data, x='date', y='value', title='Fred SP500 Data')
    ),

    dcc.Graph(
        id='yahoo-graph',
        figure=px.line(yahoo_data, x='Date', y='Close', title='Yahoo Finance AAPL Data')
    ),
])

if __name__ == '__main__':
    app.run_server(debug=True)
