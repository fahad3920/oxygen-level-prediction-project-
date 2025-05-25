import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 1. Simulated wearable sensor dataset with body parts
np.random.seed(42)
n_samples = 1000
body_parts = ['finger', 'wrist', 'earlobe']
data = {
    'heart_rate': np.random.normal(75, 10, n_samples),
    'respiration_rate': np.random.normal(16, 3, n_samples),
    'body_temp': np.random.normal(98.6, 0.7, n_samples),
    'systolic_bp': np.random.normal(120, 15, n_samples),
    'diastolic_bp': np.random.normal(80, 10, n_samples),
    'body_part': np.random.choice(body_parts, n_samples)
}

# 2. Simulate oxygen level (SpO2) based on input features
data['spo2'] = (
    100
    - 0.05 * (data['heart_rate'] - 75)
    - 0.1 * (data['respiration_rate'] - 16)
    - 0.3 * (data['body_temp'] - 98.6)
    - 0.02 * (data['systolic_bp'] - 120)
    - 0.015 * (data['diastolic_bp'] - 80)
    + np.random.normal(0, 1, n_samples)  # noise
)

df = pd.DataFrame(data)

# 3. Features and labels
X = df.drop('spo2', axis=1)
y = df['spo2']

# 4. One-hot encode the categorical 'body_part' feature
encoder = OneHotEncoder(sparse_output=False)
body_part_encoded = encoder.fit_transform(X[['body_part']])
body_part_df = pd.DataFrame(body_part_encoded, columns=encoder.get_feature_names_out(['body_part']))
X = X.drop('body_part', axis=1)
X = pd.concat([X.reset_index(drop=True), body_part_df.reset_index(drop=True)], axis=1)

# 5. Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Scale features (excluding one-hot encoded columns)
num_features = ['heart_rate', 'respiration_rate', 'body_temp', 'systolic_bp', 'diastolic_bp']
scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train[num_features])
X_test_num = scaler.transform(X_test[num_features])

# Combine scaled numerical features with one-hot encoded categorical features
X_train_scaled = np.hstack([X_train_num, X_train.drop(num_features, axis=1).values])
X_test_scaled = np.hstack([X_test_num, X_test.drop(num_features, axis=1).values])

# 7. Hyperparameter tuning for RandomForestRegressor
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

rf = RandomForestRegressor(random_state=42)
random_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=10, cv=3, verbose=1, random_state=42, n_jobs=-1)
random_search.fit(X_train_scaled, y_train)
best_rf = random_search.best_estimator_

# 8. Train GradientBoostingRegressor for comparison
gbr = GradientBoostingRegressor(random_state=42)
gbr.fit(X_train_scaled, y_train)

# 9. Predict and evaluate RandomForestRegressor
y_pred_rf = best_rf.predict(X_test_scaled)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

# 10. Predict and evaluate GradientBoostingRegressor
y_pred_gbr = gbr.predict(X_test_scaled)
rmse_gbr = np.sqrt(mean_squared_error(y_test, y_pred_gbr))
r2_gbr = r2_score(y_test, y_pred_gbr)

print(f"Random Forest RMSE: {rmse_rf:.2f}")
print(f"Random Forest R² Score: {r2_rf:.2f}")
print(f"Gradient Boosting RMSE: {rmse_gbr:.2f}")
print(f"Gradient Boosting R² Score: {r2_gbr:.2f}")

# 11. Animated plot of oxygen levels over time for sample 50
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlim(0, 50)
ax.set_ylim(min(min(y_test.values[:50]), min(y_pred_rf[:50]), min(y_pred_gbr[:50])) - 1,
            max(max(y_test.values[:50]), max(y_pred_rf[:50]), max(y_pred_gbr[:50])) + 1)
ax.set_xlabel('Sample Index')
ax.set_ylabel('Oxygen Level (%)')
ax.set_title('Animated SpO2 Prediction vs True Values (Sample 50)')
ax.grid(True)

line_true, = ax.plot([], [], label='True SpO2', color='blue', linewidth=2)
line_rf, = ax.plot([], [], label='Predicted SpO2 (RF)', color='green', linestyle='--', linewidth=2)
line_gbr, = ax.plot([], [], label='Predicted SpO2 (GBR)', color='red', linestyle=':', linewidth=2)
ax.legend()

def init():
    line_true.set_data([], [])
    line_rf.set_data([], [])
    line_gbr.set_data([], [])
    return line_true, line_rf, line_gbr

def animate(i):
    x = np.arange(i+1)
    line_true.set_data(x, y_test.values[:i+1])
    line_rf.set_data(x, y_pred_rf[:i+1])
    line_gbr.set_data(x, y_pred_gbr[:i+1])
    return line_true, line_rf, line_gbr

ani = animation.FuncAnimation(fig, animate, frames=50, init_func=init, blit=True, interval=100, repeat=False)
plt.show()
