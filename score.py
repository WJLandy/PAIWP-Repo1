import json
import numpy as np
import os
from azureml.core.model import Model
import joblib

def init():
    global model
    model_path = Model.get_model_path('model.pkl')
    model = joblib.load(model_path)

def run(raw_data):
    data = np.array(json.loads(raw_data)['data'])
    predictions = model.predict(data)
    return predictions.tolist()
