# predict_with_saved_model.py

import pickle
import pandas as pd

def load_pickle_model(path="prophet_model.pkl"):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["fe"], data["model"]

def forecast_future(model, periods=30):
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast[["ds", "yhat"]]
