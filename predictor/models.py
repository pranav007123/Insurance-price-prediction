

# Create your models here.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import os
import joblib

# Load dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'insurance.csv'))

# Prepare data
X = df[['age', 'bmi', 'children']]  # Independent variables
y = df['charges']  # Dependent variable

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, os.path.join(BASE_DIR, 'insurance_model.pkl'))
