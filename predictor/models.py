import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import joblib

# Define the base directory and data path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, 'data', 'insurance.csv')

# Load dataset
df = pd.read_csv(data_path)

# Handle missing values
df.fillna(df.median(), inplace=True)

# Separate predictors and target
X = df[['age', 'bmi', 'children']]
y = df['charges']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Save the model and scaler
model_path = os.path.join(BASE_DIR, 'insurance_model.pkl')
scaler_path = os.path.join(BASE_DIR, 'scaler.pkl')
joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)

print("Data preprocessing completed, and model saved successfully.")
