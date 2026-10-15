
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Load data
data = pd.read_csv("data/tcs_stock_data.csv")
data["Date"] = pd.to_datetime(data["Date"], utc=True)
data = data.sort_values("Date").reset_index(drop=True)

# 2. Create the target: next trading day's Close
data["Next_Close"] = data["Close"].shift(-1)
data = data.dropna(subset=["Next_Close"]).copy()

# 3. Select multiple input features
features = ["Open", "High", "Low", "Close", "Volume"]

X = data[features]
y = data["Next_Close"]

# 4. Chronological 80/20 split
split_index = int(len(data) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Input features:", features)

# 5. Train Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Predict
y_pred = model.predict(X_test)


# STEP 7: ESTIMATE THE NEXT TRADING DAY'S CLOSE

# Get the latest available trading record
latest_row = data.iloc[[-1]]

# Select the same features used during training
latest_features = latest_row[
    ["Open", "High", "Low", "Close", "Volume"]
]

# Generate the estimate
next_close_estimate = model.predict(latest_features)[0]

print("\n========== NEXT TRADING DAY ESTIMATE ==========")
print("Latest available date:", latest_row["Date"].iloc[0])
print(
    "Latest closing price: Rs.",
    round(latest_row["Close"].iloc[0], 2)
)
print(
    "Estimated next trading day close: Rs.",
    round(next_close_estimate, 2)
)


# 7. Baseline: next Close = current Close
baseline_pred = X_test["Close"].to_numpy()

# 8. Evaluate
model_mae = mean_absolute_error(y_test, y_pred)
model_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

baseline_mae = mean_absolute_error(y_test, baseline_pred)
baseline_rmse = np.sqrt(
    mean_squared_error(y_test, baseline_pred)
)

print("\n--- Multiple-Feature Model ---")
print(f"MAE: Rs. {model_mae:.2f}")
print(f"RMSE: Rs. {model_rmse:.2f}")

print("\n--- Baseline ---")
print(f"MAE: Rs. {baseline_mae:.2f}")
print(f"RMSE: Rs. {baseline_rmse:.2f}")

# 9. Compare predictions
results = pd.DataFrame({
    "Date": data["Date"].iloc[split_index:].values,
    "Actual_Close": y_test.to_numpy(),
    "Predicted_Close": y_pred
})

print("\n--- First 10 Predictions ---")
print(
    results.head(10).assign(
        Actual_Close=lambda df: df["Actual_Close"].round(2),
        Predicted_Close=lambda df: df["Predicted_Close"].round(2)
    ).to_string(index=False)
)

# 10. Plot actual vs predicted
plt.figure(figsize=(12, 6))

plt.plot(
    results["Date"],
    results["Actual_Close"],
    label="Actual Closing Price"
)

plt.plot(
    results["Date"],
    results["Predicted_Close"],
    label="Predicted Closing Price"
)

plt.title("TCS Stock Prediction - Multiple Features")
plt.xlabel("Date")
plt.ylabel("Closing Price (INR)")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("tcs_actual_vs_predicted.png", dpi=300)
print("\nGraph saved as tcs_actual_vs_predicted.png")

plt.show()
