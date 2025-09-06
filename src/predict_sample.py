import pandas as pd
from catboost import CatBoostRegressor
import joblib
from src.preprocessing import preprocess_data

# Load model and encoders
model = joblib.load("models/catboost_model.pkl")
encoder = joblib.load('models/target_encoder.pkl')
ohe = joblib.load('models/ohe_encoder.pkl')
pt = joblib.load('models/pt.pkl')
pt_y = joblib.load('models/pt_y.pkl')

sample_dict = {
    "Crop": "Dry chillies",
    "Season": "Whole Year",
    "State": "Meghalaya",
    "Area": 1766,
    "Production": 1106,
    "Annual_Rainfall": 3818.2,
    "Fertilizer": 168070.22,
    "Pesticide": 547.46,
}

# Convert to DataFrame
sample_df = pd.DataFrame([sample_dict])

# Preprocess
sample_df = preprocess_data(sample_df=sample_df, is_sample=True, encoder=encoder, ohe=ohe, pt=pt)
# Predict
pred = model.predict(sample_df)
pred = pt_y.inverse_transform(pred.reshape(-1,1))
print('Predicted Yield:', pred[0][0])
