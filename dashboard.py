import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime, timedelta

# =====================
# Load Dataset
# =====================
df = pd.read_csv("solar_data_khulna_from_jan_2014_to_nov_2022.csv.zip")

# =====================
# Train Random Forest
# =====================
X = df[["Irradiance", "Hour", "Month", "Day", "Year"]]
y = df["Temperature"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

pred = model.predict(X_test)
mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred) * 100

# =====================
# Last 30 Days Report
# =====================
# Create Date column
df["Date"] = pd.to_datetime(df[["Year", "Month", "Day"]])

# Daily average temperature
daily_avg = df.groupby("Date")["Temperature"].mean().reset_index()

# Last 30 days
last_30 = daily_avg.tail(30)

last30_fig = px.line(
    last_30,
    x="Date",
    y="Temperature",
    title="Last 30 Days Average Temperature",
    markers=True
)

# =====================
# Tomorrow Hourly Prediction
# =====================
tomorrow = datetime.now() + timedelta(days=1)
hours = list(range(6, 19))
predicted_temps = []

for h in hours:
    input_df = pd.DataFrame([{
        "Irradiance": 600,
        "Hour": h,
        "Month": tomorrow.month,
        "Day": tomorrow.day,
        "Year": tomorrow.year
    }])
    predicted_temps.append(model.predict(input_df)[0])

tomorrow_fig = px.line(
    x=hours,
    y=predicted_temps,
    labels={"x": "Hour", "y": "Temperature (°C)"},
    title="Tomorrow Hourly Temperature Prediction"
)

# =====================
# Monthly Average Bar
# =====================
monthly_avg = df.groupby("Month")["Temperature"].mean().reset_index()
monthly_fig = px.bar(
    monthly_avg,
    x="Month",
    y="Temperature",
    title="Monthly Average Temperature"
)

# =====================
# Accuracy Comparison
# =====================
accuracy_fig = px.bar(
    x=["Random Forest"],
    y=[r2],
    labels={"x": "Model", "y": "Accuracy (%)"},
    title="Model Accuracy",
    text=[f"{r2:.2f}%"]
)
accuracy_fig.update_yaxes(range=[0, 100])


# =====================
# HTML + CSS Dashboard
# =====================
html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Temperature Prediction Dashboard</title>

<style>
body {{
    margin: 0;
    font-family: Arial, sans-serif;
    background: #f4f6f8;
}}

.header {{
    background: #1f2937;
    color: white;
    padding: 15px;
    text-align: center;
}}

.container {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 15px;
    padding: 15px;
}}

.card {{
    background: white;
    border-radius: 10px;
    padding: 12px;
    min-height: 420px;
}}

.full {{
    grid-column: span 2;
}}
</style>
</head>

<body>

<div class="header">
    <h2>Temperature Prediction Dashboard</h2>
    <p>MAE: {mae:.2f} °C | Accuracy (R²): {r2:.2f} %</p>
</div>

<div class="container">

<div class="card">
{monthly_fig.to_html(full_html=False, include_plotlyjs='cdn')}
</div>

<div class="card">
{accuracy_fig.to_html(full_html=False, include_plotlyjs=False)}
</div>

<div class="card full">
{last30_fig.to_html(full_html=False, include_plotlyjs=False)}
</div>

<div class="card full">
{tomorrow_fig.to_html(full_html=False, include_plotlyjs=False)}
</div>

</div>

</body>
</html>
"""

with open("temperature_dashboard_all_in_one.html", "w", encoding="utf-8") as f:
    f.write(html)

print("Dashboard created: temperature_dashboard_all_in_one.html")
