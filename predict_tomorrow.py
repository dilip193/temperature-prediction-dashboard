import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

# Load data
df = pd.read_csv("solar_data_khulna_from_jan_2014_to_nov_2022.csv.zip")

# Train model
X = df[["Irradiance", "Hour", "Month", "Day", "Year"]]
y = df["Temperature"]

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
model.fit(X, y)

# ---- Tomorrow prediction ----
tomorrow = datetime.now() + timedelta(days=1)

# Example values (you can change hour & irradiance)
hour = 12          # noon
irradiance = 600   # estimated irradiance

input_data = [[
    irradiance,
    hour,
    tomorrow.month,
    tomorrow.day,
    tomorrow.year
]]

predicted_temp = model.predict(input_data)

print(
    f"Predicted temperature for tomorrow "
    f"({tomorrow.date()} at {hour}:00): "
    f"{predicted_temp[0]:.2f} °C"
)
