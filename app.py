import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from datetime import datetime, timedelta
import joblib
import requests

# Load the Decision Tree model
model = joblib.load('model.pkl')

# Charger les données
df = pd.read_csv('Base_13.csv')
df['date'] = pd.to_datetime(df['date'])
df['Days_Until_Maintenance'] = (df['date'] - pd.Timestamp.now()).dt.days

print("Modèle sauvegardé dans model.pkl")
# Vérifiez que les colonnes nécessaires existent dans le DataFrame
# Assurez-vous que les colonnes 'pressure', 'flow_in', 'flow_out', 'conductivity' existent dans le CSV
required_columns = ['pressure', 'flow_rate_in', 'flow_rate_out', 'conductivity', 'date']
if not all(col in df.columns for col in required_columns):
    st.error("Le fichier CSV doit contenir les colonnes : 'pressure', 'flow_rate_in', 'flow_rate_out', 'conductivity' et 'date'.")
else:
    # Préparer les données pour l'entraînement
    X = df[['pressure', 'flow_rate_in', 'flow_rate_out', 'conductivity']]
    y = df['Days_Until_Maintenance']  # Variable cible

    # Créer et entraîner le modèle
    model = DecisionTreeRegressor()
    model.fit(X, y)

    # Interface Streamlit
    st.title("Simulation of the predictive prediction of the Osmosis system")

    # Entrées utilisateur
    pressure = st.number_input('Pression', value=0.0)
    flow_rate_in = st.number_input('Flow Input', value=0.0)
    flow_rate_out = st.number_input('Flow Output', value=0.0)
    conductivity = st.number_input('Conductivity', value=0.0)
    api_url = "http://localhost:8000/predict"
api_url = "http://localhost:8000/predict"

if st.button('Prédire'):
    # Préparer les données pour l'API
    data = {
        "pressure": pressure,
        "flow_rate_in": flow_rate_in,
        "flow_rate_out": flow_rate_out,
        "conductivity": conductivity
    }
    
    # Envoyer une requête POST à l'API
    response = requests.post(api_url, json=data)
    
    if response.status_code == 200:
        result = response.json()
        st.success(f"La prochaine maintenance sera le : {result['predicted_date']} (dans {result['predicted_days']} jours)")
    else:
        st.error(f"Erreur lors de la prédiction : {response.text}")


    
    if st.button('Prédire'):
        input_data = np.array([[pressure, flow_rate_in, flow_rate_out, conductivity]])
        predicted_days = model.predict(input_data)[0]
        predicted_date = datetime.now() + timedelta(days=predicted_days)
        st.success(f"The next maintenance will be for the date: : {predicted_date.strftime('%Y-%m-%d')}")
 
joblib.dump(model, 'model.pkl')

print("Modèle sauvegardé dans model.pkl")
