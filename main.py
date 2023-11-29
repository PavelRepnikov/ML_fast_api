from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
import pickle
import sklearn

app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str 
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


loaded_model = pickle.load(open('finalized_model.pickle', 'rb'))


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    item.mileage = pd.to_numeric(item.mileage.replace(' kmpl', '').replace(' km/kg', ''), errors='coerce')
    item.engine = pd.to_numeric(item.engine.replace(' CC', ''), errors='coerce')
    item.max_power = pd.to_numeric(item.max_power.replace(' bhp', ''), errors='coerce')
    item.engine = int(item.engine)
    item.seats = int(item.seats)

    input_data = [item.year, item.km_driven, item.mileage, item.engine, item.max_power, item.seats]
    result = loaded_model.predict([input_data])
    return result[0]


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[dict]:
    result_list = []
    for item in items:

        item.mileage = pd.to_numeric(item.mileage.replace(' kmpl', '').replace(' km/kg', ''), errors='coerce')
        item.engine = pd.to_numeric(item.engine.replace(' CC', ''), errors='coerce')
        item.max_power = pd.to_numeric(item.max_power.replace(' bhp', ''), errors='coerce')
        item.engine = int(item.engine)
        item.seats = int(item.seats)

        input_data = [
            item.year,
            item.km_driven,
            item.mileage,
            item.engine,
            item.max_power,
            item.seats,

        ]
        result = loaded_model.predict([input_data])
        result_dict = item.dict()  # Convert Item to dictionary
        result_dict['prediction'] = result[0]
        result_list.append(result_dict)

    return result_list

