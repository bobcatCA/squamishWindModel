"""
mudelTesting.py
Purpose is to test the Vanilla Net model, plot vs. actual/measured
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load your DataFrame
df = pd.read_csv('mergedOnSpeed.csv')
df['datetime'] = pd.to_datetime(df['datetime'])

# Split data into features and target
X = df.drop(columns=['speed', 'datetime'])
y = df['speed']

# Load the pre-trained model
model = load_model('daytimeTempsHumSkyPressSpeed.keras')

# Normalize features
scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X)

# Run model to predict
y_pred = model.predict(X_test_scaled)

# Now do the plotting
x_plot = df['datetime']
fig, ax = plt.subplots()
ax.plot(x_plot, y, label='Measured')
ax.plot(x_plot, y_pred, label='Predicted')
plt.ylabel('Wind Speed (knots)')
plt.legend()
plt.show()
