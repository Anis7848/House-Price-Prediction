from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Load dataset
df = pd.read_csv("C:\\Users\\ANIS\\Downloads\\Machine Learning\\House Price Prediction\\house_prices.csv")

# Encode categorical features
label_encoders = {}
for col in ['district', 'tahsil', 'village', 'bedrooms']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features and target
X = df[['district', 'tahsil', 'village', 'house_area', 'bedrooms']]
y = df['price']

# Train model
model = RandomForestRegressor()
model.fit(X, y)

import joblib

joblib.dump(model, 'model.pkl')
joblib.dump(label_encoders, 'encoders.pkl')