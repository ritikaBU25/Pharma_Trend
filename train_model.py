import pickle
import pandas as pd
from prophet import Prophet
from feature_pipeline import FeatureEngineer

def train_and_save_model(input_csv="training_data.csv",
                         model_path="prophet_model.pkl",
                         date_col=None, value_col=None):

    df = pd.read_csv(input_csv)

    if date_col is None:
        date_col = df.columns[0]
    if value_col is None:
        value_col = df.columns[1]

    fe = FeatureEngineer(date_col=date_col, value_col=value_col)
    df_fe = fe.transform(df)

    model = Prophet(weekly_seasonality=True, yearly_seasonality=True)
    model.fit(df_fe)

    with open(model_path, "wb") as f:
        pickle.dump({"fe": fe, "model": model}, f)

    print(f"Model saved → {model_path}")

if __name__ == "__main__":
    train_and_save_model()
