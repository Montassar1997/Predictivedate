from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
from datetime import datetime, timedelta

# Définir le modèle de données pour les requêtes
class PredictionRequest(BaseModel):
    pressure: float
    flow_rate_in: float
    flow_rate_out: float
    conductivity: float

# Initialiser l'application FastAPI
app = FastAPI()

# Charger le modèle une seule fois au démarrage
model = joblib.load('model.pkl')

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        # Préparer les données pour la prédiction
        input_data = np.array([[request.pressure, request.flow_rate_in, request.flow_rate_out, request.conductivity]])
        predicted_days = model.predict(input_data)[0]
        predicted_date = datetime.now() + timedelta(days=predicted_days)
        
        # Retourner le résultat en JSON
        return {
            'predicted_days': predicted_days,
            'predicted_date': predicted_date.strftime('%Y-%m-%d')
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
