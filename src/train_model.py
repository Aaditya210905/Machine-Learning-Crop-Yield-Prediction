import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from src.preprocessing import preprocess_data
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

df = pd.read_csv('data/crop_yield.csv')

X = df.drop(columns=['Yield'])
y = df['Yield']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess data
X_train, X_test, y_train, y_test, encoder,ohe, pt, pt_y, cat_cols, num_cols = preprocess_data(X_train, X_test, y_train, y_test)

# Train CatBoost
model = CatBoostRegressor(task_type="GPU",depth=8,iterations=1500,learning_rate=0.1, verbose=0)
model.fit(X_train, y_train)

#Create a Function to Evaluate Model
def evaluate_model(true, predicted):
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mean_squared_error(true, predicted))
    r2_square = r2_score(true, predicted)
    return mae, rmse, r2_square

y_pred = model.predict(X_test)
mae, rmse, r2 = evaluate_model(y_test, y_pred)
print(f'MAE: {mae}, RMSE: {rmse}, R2: {r2}')

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Save trained model and encoders
joblib.dump(model, "models/catboost_model.pkl")
joblib.dump(encoder, 'models/target_encoder.pkl')
joblib.dump(ohe, 'models/ohe_encoder.pkl')
joblib.dump(pt, 'models/pt.pkl')
joblib.dump(pt_y, 'models/pt_y.pkl')
