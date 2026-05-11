from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from enum import Enum
import pickle
import pandas as pd

app = FastAPI(title="Car Price Prediction API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


with open('artifacts/model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('artifacts/feature_columns.pkl', 'rb') as f:
    feature_columns = pickle.load(f)


class BrandEnum(str, Enum):
    Audi     = "Audi"
    BMW      = "BMW"
    Ford     = "Ford"
    Honda    = "Honda"
    Mercedes = "Mercedes"
    Tesla    = "Tesla"
    Toyota   = "Toyota"

class FuelTypeEnum(str, Enum):
    Petrol   = "Petrol"
    Diesel   = "Diesel"
    Electric = "Electric"
    Hybrid   = "Hybrid"

class TransmissionEnum(str, Enum):
    Automatic = "Automatic"
    Manual    = "Manual"

class ConditionEnum(str, Enum):
    New      = "New"
    Used     = "Used"
    Like_New = "Like New"


class CarFeatures(BaseModel):
    Brand       : BrandEnum
    Year        : int
    Engine_Size : float
    Fuel_Type   : FuelTypeEnum
    Transmission: TransmissionEnum
    Mileage     : int
    Condition   : ConditionEnum

@app.get("/")
def home():
    return {"message": "Car Price Prediction API is running"}

@app.post("/predict")
def predict(car: CarFeatures):
    input_dict = {
        'Year'        : car.Year,
        'Engine Size' : car.Engine_Size,
        'Mileage'     : car.Mileage,
        'Brand'       : car.Brand.value,
        'Fuel Type'   : car.Fuel_Type.value,
        'Transmission': car.Transmission.value,
        'Condition'   : car.Condition.value
    }
    input_df = pd.DataFrame([input_dict])

    
    input_encoded = pd.get_dummies(input_df, columns=['Brand', 'Fuel Type', 'Transmission', 'Condition'])

    
    for col in feature_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    input_encoded = input_encoded[feature_columns]

    predicted_price = model.predict(input_encoded)[0]

    return {
        "predicted_price": round(float(predicted_price), 2),
        "message"        : f"Estimated car price is ${predicted_price:,.2f}"
    }