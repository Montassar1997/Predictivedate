from flask import Flask, jsonify, request
# from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta

# Définir le modèle de données pour les requêtes
# class PredictionRequest(BaseModel):
    # pressure: float
    # flow_rate_in: float
    # flow_rate_out: float
    # conductivity: float

# Initialiser l'application FastAPI
app = Flask('__name__')

# Charger le modèle une seule fois au démarrage
model = joblib.load('decision_tree_regressor.pkl')

@app.route("/predict", methods=['POST'])
def predict():
    try:
        # Préparer les données pour la prédiction
        data=request.json
        input_data = np.array([[data['pressure'], data['flow_rate_in'], data['flow_rate_out'], data['conductivity']]])
        days_until_maintenance = model.predict(input_data)[0]
        date = pd.Timestamp.now() + timedelta(days=days_until_maintenance)
        # Retourner le résultat en JSON
        return jsonify({
            'predicted_date': date,
        })
    except Exception as e:
        print(e)

if __name__=='__main__':
	app.run(port=8080)
