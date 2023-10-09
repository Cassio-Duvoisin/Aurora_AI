
import lightgbm as lgb
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Load the processed data
dados = pd.read_csv('processed_data.csv')

# Separate the target column from the features
X = dados.iloc[:, :-1]
y = dados.iloc[:, -1]

# Initialize the normalizer
scaler = MinMaxScaler()

# Assuming the scaler was fitted on the same training data as before
# Transform the data using the same scaler (Note: we don't fit on the new data!)
X_normalized = scaler.transform(X)

# Load the trained model
lgbm_model_loaded = lgb.Booster(model_file='lgbm_model.txt')

# Predict using the loaded model
y_pred = lgbm_model_loaded.predict(X_normalized)

# Evaluate the model's performance using RMSE
mse = mean_squared_error(y, y_pred)
rmse = mse ** 0.5

# Approximate accuracy metric
resolution = 0.003
accurate_predictions = sum(abs(y - y_pred) < resolution)
approx_accuracy = accurate_predictions / len(y)

# Return to the original proportion
resolution = resolution * 400

# Display the model's accuracy 
print(f"Accuracy: {approx_accuracy} for {resolution} resolution")
