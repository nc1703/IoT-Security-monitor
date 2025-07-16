"""
IoT Security Monitoring System
------------------------------

This script simulates a basic IoT security monitoring system that collects device data
(e.g., temperature, motion, signal strength) and uses anomaly detection to flag potentially
malicious activity or system malfunction. It uses an unsupervised machine learning algorithm
(Isolation Forest) to detect anomalies without needing labeled data.

Output:
- Anomaly detection printed to console
- Model and scaler saved for future inference

Note: This is a simulated dataset. In production, integrate with real sensor data via MQTT, APIs, etc.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

# ------------------------------
# Step 1: Simulate IoT sensor data
# ------------------------------
# Each row simulates a snapshot of sensor readings from an IoT device
np.random.seed(42)
normal_data = {
    'temperature': np.random.normal(loc=22, scale=2, size=100),
    'humidity': np.random.normal(loc=45, scale=5, size=100),
    'motion': np.random.poisson(lam=2, size=100),  # e.g., motion sensor events
    'signal_strength': np.random.normal(loc=-70, scale=5, size=100)
}

# Inject some anomalies
anomalies = {
    'temperature': [60, -10],  # extreme temps
    'humidity': [10, 90],      # extreme humidity
    'motion': [15, 20],        # very active motion
    'signal_strength': [-30, -120]  # abnormal signal
}

# Convert to DataFrame
df_normal = pd.DataFrame(normal_data)
df_anomaly = pd.DataFrame(anomalies)
df = pd.concat([df_normal, df_anomaly], ignore_index=True)

# ------------------------------
# Step 2: Preprocess and scale data
# ------------------------------
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# ------------------------------
# Step 3: Train Isolation Forest for anomaly detection
# ------------------------------
model = IsolationForest(contamination=0.05, random_state=42)
model.fit(scaled_data)

# Predict anomalies (-1 = anomaly, 1 = normal)
df['anomaly'] = model.predict(scaled_data)

# Display anomalies
print("Detected Anomalies:")
print(df[df['anomaly'] == -1])

# ------------------------------
# Step 4: Save model and scaler
# ------------------------------
joblib.dump(model, 'iot_model.pkl')
joblib.dump(scaler, 'iot_scaler.pkl')
