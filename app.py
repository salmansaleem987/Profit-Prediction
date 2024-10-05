from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import joblib

# Create the FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins or specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Load the saved linear regression model
model = joblib.load('linear_regression_model.pkl')

# Define a BaseModel for input data
class ProfitPredictionInput(BaseModel):
    rd_spend: float
    administration: float
    marketing_spend: float
    state: str

# A dictionary for one-hot encoding states
state_dict = {'New York': [1, 0, 0], 'California': [0, 1, 0], 'Florida': [0, 0, 1]}

# Prediction route
@app.post("/predict-profit")
def predict_profit(input_data: ProfitPredictionInput):
    # One-hot encode the state
    if input_data.state in state_dict:
        state_encoded = state_dict[input_data.state]
    else:
        return {"error": "Invalid state. Choose from 'New York', 'California', or 'Florida'."}

    # Combine inputs into a feature array
    features = np.array([[input_data.rd_spend, input_data.administration, input_data.marketing_spend] + state_encoded])

    # Predict the profit using the loaded model
    predicted_profit = model.predict(features)

    # Return the predicted profit
    return {"Predicted Profit": round(predicted_profit[0], 2)}
