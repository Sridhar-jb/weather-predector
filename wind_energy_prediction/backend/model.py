import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

print("STARTING MODEL TRAINING...")

print("Current working directory:", os.getcwd())

# Load dataset
data = pd.read_csv("dataset.csv")
print("Dataset loaded")

X = data.drop("EnergyOutput", axis=1)
y = data["EnergyOutput"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

model = LinearRegression()
model.fit(X_train, y_train)

print("Model trained")

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("MODEL FILES SAVED SUCCESSFULLY")
