# train_model.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

# Load dataset
df = pd.read_csv("AirQuality.csvst")

# Drop unnamed columns and missing values
df = df.dropna(axis=1, how="all")
df = df.dropna()

# Target and features
y = df["CO(GT)"]
X = df.drop(["CO(GT)", "Date", "Time"], axis=1, errors="ignore")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost
model = XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "xgb_model.pkl")

# Predictions
y_pred = model.predict(X_test)

# Evaluation plots
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual CO(GT)")
plt.ylabel("Predicted CO(GT)")
plt.title("Actual vs Predicted")
plt.savefig("actual_vs_predicted.png")
plt.close()

residuals = y_test - y_pred
plt.hist(residuals, bins=30, edgecolor="black")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Residuals Histogram")
plt.savefig("residuals_hist.png")
plt.close()

importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(8,5))
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), X.columns[indices], rotation=90)
plt.title("Feature Importances")
plt.tight_layout()
plt.savefig("feature_importances.png")
plt.close()

print("✅ Model trained and saved as xgb_model.pkl")
