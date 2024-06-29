import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Generate synthetic data
np.random.seed(42)
heights = np.random.uniform(1.5, 2.0, 1000)  # heights in meters
weights = np.random.uniform(50, 100, 1000)   # weights in kilograms
bmis = weights / (heights ** 2)

# Create a DataFrame
data = pd.DataFrame({
    'Height': heights,
    'Weight': weights,
    'BMI': bmis
})

data.to_csv('bmi_data.csv', index=False)

# Split the data into features and target variable
X = data[['Height', 'Weight']]
y = data['BMI']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Train the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Save the models
joblib.dump(lr_model, 'lr_bmi_predictor.pkl')
joblib.dump(rf_model, 'rf_bmi_predictor.pkl')

# Compute metrics
lr_y_pred = lr_model.predict(X_test)
rf_y_pred = rf_model.predict(X_test)

lr_mae = mean_absolute_error(y_test, lr_y_pred)
lr_r2 = r2_score(y_test, lr_y_pred)

rf_mae = mean_absolute_error(y_test, rf_y_pred)
rf_r2 = r2_score(y_test, rf_y_pred)

metrics = {
    "Linear Regression": {"MAE": lr_mae, "R2": lr_r2},
    "Random Forest": {"MAE": rf_mae, "R2": rf_r2}
}

# Save the metrics
joblib.dump(metrics, 'metrics.pkl')
