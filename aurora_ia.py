import lightgbm as lgb
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

resolution = 0.00375

# Load the processed data
dados = pd.read_csv('processed_data.csv')

# Separate the target column from the features
X = dados.iloc[:, :-1]
y = dados.iloc[:, -1]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the normalizer
scaler = MinMaxScaler()

# Fit and transform the training data
X_train_normalized = scaler.fit_transform(X_train)

# Transform the test data (Note: we don't fit on the test data!)
X_test_normalized = scaler.transform(X_test)

# Create a LightGBM dataset
train_data = lgb.Dataset(X_train_normalized, label=y_train)

# Set the model parameters
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 7500,
    'learning_rate': 0.094,
    'feature_fraction': 0.9,
    'force_row_wise': True
}

# Train the model
num_round = 100
lgbm_model = lgb.train(params, train_data, num_round)

# Predict on the test set
y_pred_lgbm = lgbm_model.predict(X_test_normalized)

# Save the trained model
lgbm_model.save_model('lgbm_model.txt')

# Evaluate the model's performance using RMSE
mse_lgbm = mean_squared_error(y_test, y_pred_lgbm)
rmse_lgbm = mse_lgbm ** 0.5

# Approximate accuracy metric
accurate_predictions_lgbm = sum(abs(y_test - y_pred_lgbm) < resolution)
approx_accuracy_lgbm = accurate_predictions_lgbm / len(y_test)

# Return to the original proportion
resolution = resolution * 400

# Model's trained accuracy 
print(f"Accuracy: {approx_accuracy_lgbm} for {resolution} resolution")

