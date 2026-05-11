import pickle
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from build_pipeline import build_pipeline


df = pd.read_csv('data/car_price_prediction_.csv')
print("Dataset loaded:", df.shape)

df = df.drop(columns=['Car ID', 'Model'])

df = pd.get_dummies(df, columns=['Brand', 'Fuel Type', 'Transmission', 'Condition'])

feature_columns = [col for col in df.columns if col != 'Price']

X = df[feature_columns]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = build_pipeline()
pipeline.fit(X_train, y_train)

os.makedirs('artifacts', exist_ok=True)

with open('artifacts/model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

with open('artifacts/feature_columns.pkl', 'wb') as f:
    pickle.dump(feature_columns, f)

print("Model saved!")

y_pred = pipeline.predict(X_test)
print("MAE :", round(mean_absolute_error(y_test, y_pred), 2))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, y_pred)), 2))
print("R2  :", round(r2_score(y_test, y_pred), 4))