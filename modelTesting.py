import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load your DataFrame
df = pd.read_csv('mergedOnSpeed.csv')
df['datetime'] = pd.to_datetime(df['datetime'])

# Split data into features and target
X = df.drop(columns=['speed', 'datetime'])
y = df['speed']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)
model = load_model('daytimeTempsHumSkyPressSpeed.keras')

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

y_pred = model.predict(X_test_scaled)

testIndStart = X_test.index.values[0]
testIndEnd = X_test.index.values[-1]

x_plot = df['datetime'].iloc[testIndStart:testIndEnd + 1]

fig, ax = plt.subplots()
ax.plot(x_plot, y_test, label='Measured')
ax.plot(x_plot, y_pred, label='Predicted')
plt.ylabel('Wind Speed (knots)')
plt.legend()
plt.show()
