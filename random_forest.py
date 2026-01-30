import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime, timedelta

# =========================
# Load Dataset
# =========================
df = pd.read_csv(
    "solar_data_khulna_from_jan_2014_to_nov_2022.csv.zip"
)

# =========================
# Features and Target
# =========================
X = df[["Irradiance", "Hour", "Month", "Day", "Year"]]
y = df["Temperature"]

# =========================
# Train-Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# Random Forest Model
# =========================
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# =========================
# Predictions
# =========================
pred = model.predict(X_test)

# =========================
# Evaluation Metrics
# =========================
mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)

print("Random Forest Regression Results")
print("--------------------------------")
print(f"Mean Absolute Error (MAE): {mae:.2f} °C")
print(f"Accuracy (R² Score): {r2 * 100:.2f} %")

# =========================
# Feature Importance
# =========================
importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nFeature Importance:")
print(importance)

# =========================
# Tomorrow Temperature Prediction
# =========================
tomorrow = datetime.now() + timedelta(days=1)

# Example assumptions (can be changed)
hour = 12            # noon
irradiance = 600     # estimated irradiance

input_data = [[
    irradiance,
    hour,
    tomorrow.month,
    tomorrow.day,
    tomorrow.year
]]

tomorrow_temp = model.predict(input_data)

print(
    f"\nPredicted temperature for {tomorrow.date()} "
    f"at {hour}:00 = {tomorrow_temp[0]:.2f} °C"
)
