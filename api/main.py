import os
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Union, Type, Literal
import joblib
import uvicorn
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

app = FastAPI()

# Pydantic model for input data
class PredictionInput(BaseModel):
    modelName: Literal["Linear Regression", "Decision Tree", "Random Forest", "Gradient Descent"]
    tv: float

# Supported model types
SupportedModel = Union[Type[LinearRegression], Type[DecisionTreeRegressor], Type[RandomForestRegressor]]


# Get the current directory of the FastAPI file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define the path to the models directory
models_dir = os.path.join(current_dir, '..', 'models')

# Load the models using the updated paths
lr_model = joblib.load(os.path.join(models_dir, 'lr_model.pkl'))
dt_model = joblib.load(os.path.join(models_dir, 'dt_model.pkl'))
rf_model = joblib.load(os.path.join(models_dir, 'rf_model.pkl'))
gd_params = joblib.load(os.path.join(models_dir, 'gd_params.pkl'))

# get our trained weight, bias, standard deviation and mean
m_gd = gd_params['m_gd']
b_gd = gd_params['b_gd']
X_mean = gd_params['X_mean']
X_std = gd_params['X_std']
Y_mean = gd_params['Y_mean']
Y_std = gd_params['Y_std']

# Function to make predictions
def predict_fast_api(modelName: str, tv: float):
    # Add constraints for tv (0 <= tv <= 1000)
    if not (0 <= tv <= 1000):
        raise HTTPException(status_code=400, detail="TV marketing budget must be between 0 and 1000")

    # Get the model based on the modelName
    if modelName == "Linear Regression":
        model = lr_model
    elif modelName == "Decision Tree":
        model = dt_model
    elif modelName == "Random Forest":
        model = rf_model
    elif modelName == "Gradient Descent":
        # normalize input
        tv_norm = (tv - X_mean) / X_std
        tv_sales_norm = m_gd * tv_norm + b_gd
        # denormalize output
        tv_sales = tv_sales_norm * Y_std + Y_mean 
        return tv_sales
    else:
        allowed_models = ["Linear Regression", "Decision Tree", "Random Forest", "Gradient Descent"]
        raise HTTPException(status_code=400, detail=f"Invalid model name. Allowed models are: {', '.join(allowed_models)}")

    # Reshape tv for scikit-learn models
    tv_input = np.array([[tv]])
    tv_sales = model.predict(tv_input)[0][0] if isinstance(model, LinearRegression) else model.predict(tv_input)[0]

    return tv_sales

@app.post("/predict")
async def predict_sales(input_data: PredictionInput):
    """
    Predicts tv sales based on TV marketing budget and model name.
    """
    try:
        tv_sales_prediction = predict_fast_api(input_data.modelName, input_data.tv)
    except HTTPException as e:
        return {"error": e.detail}

    return {"tv_sales": tv_sales_prediction}

# Run the FastAPI app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)