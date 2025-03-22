import numpy as np
import pandas as pd
from models.linear_regression import LinearRegression

# Load dataset
df = pd.read_csv("data/synthetic_data.csv")
X = df.iloc[:, :-1].values  # Features
y = df.iloc[:, -1].values.reshape(-1, 1)  # Target

# Train model
model = LinearRegression(learning_rate=0.01, iterations=2000)
model.fit(X, y)

# Test Predictions
X_test = np.array([[5, 3, 4], [6, 2, 5]])  # Example test data
predictions = model.predict(X_test)
print(f"Predictions: {predictions}")
