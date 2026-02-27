import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Create Synthetic Data
np.random.seed(42)
data_size = 1000
amounts = np.random.uniform(10, 5000, data_size)
hours = np.random.randint(0, 24, data_size)

# Logic: Fraud if (high amount AND night) or (very high amount)
# This mimics real-world patterns for the ML to learn
fraud_prob = ( (amounts > 3000) * 0.7 + ((hours < 6) | (hours > 22)) * 0.3 )
labels = (fraud_prob + np.random.normal(0, 0.2, data_size) > 0.6).astype(int)

df = pd.DataFrame({'amount': amounts, 'hour': hours, 'fraud': labels})

# 2. Train Model
X = df[['amount', 'hour']]
y = df['fraud']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_scaled, y)

# 3. Save artifacts
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model trained and saved!")