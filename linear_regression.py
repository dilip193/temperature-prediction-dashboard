import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load data
df = pd.read_csv("solar_data_khulna_from_jan_2014_to_nov_2022.csv.zip")

X = df[["Irradiance", "Hour", "Month"]]
y = df["Temperature"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)

print("Linear Regression Results")
print("MAE:", round(mae, 2), "°C")
print("R² Score (Accuracy):", round(r2 * 100, 2), "%")
