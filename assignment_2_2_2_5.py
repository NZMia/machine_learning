import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load and preprocess data
def load_and_preprocess_data():
    data = pd.read_csv('Heat_Influx_insulation_east_south_north.csv')
    X = data[['Insulation', 'East', 'South', 'North']].values
    y = data['HeatFlux'].values
    
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    return X_scaled, y_scaled, scaler_y

# Load data
X, y, scaler_y = load_and_preprocess_data()

# Load saved models
adagrad_model = load_model('best_adagrad_model.h5')
sgd_model = load_model('best_heat_influx_model.h5')

# Make predictions
adagrad_pred_scaled = adagrad_model.predict(X).flatten()
sgd_pred_scaled = sgd_model.predict(X).flatten()

# Inverse transform predictions and target
target = scaler_y.inverse_transform(y.reshape(-1, 1)).flatten()
adagrad_pred = scaler_y.inverse_transform(adagrad_pred_scaled.reshape(-1, 1)).flatten()
sgd_pred = scaler_y.inverse_transform(sgd_pred_scaled.reshape(-1, 1)).flatten()

# Calculate errors
adagrad_error = target - adagrad_pred
sgd_error = target - sgd_pred

# Create two separate plots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)

# Plot for SGD model
ax1.plot(range(len(target)), target, color='blue', label='Target', alpha=0.7)
ax1.plot(range(len(sgd_pred)), sgd_pred, color='orange', label='SGD Prediction', alpha=0.7)
ax1.plot(range(len(sgd_error)), sgd_error, color='red', label='SGD Error', alpha=0.7)
ax1.set_ylabel('Heat Flux / Error')
ax1.set_title('SGD Model: Target, Prediction, and Error')
ax1.legend()
ax1.grid(True)

# Plot for Adagrad model
ax2.plot(range(len(target)), target, color='blue', label='Target', alpha=0.7)
ax2.plot(range(len(adagrad_pred)), adagrad_pred, color='orange', label='Adagrad Prediction', alpha=0.7)
ax2.plot(range(len(adagrad_error)), adagrad_error, color='red', label='Adagrad Error', alpha=0.7)
ax2.set_xlabel('Input Index')
ax2.set_ylabel('Heat Flux / Error')
ax2.set_title('Adagrad Model: Target, Prediction, and Error')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# Print some statistics
print(f"Total number of samples: {len(target)}")
print(f"Shape of the input data (X): {X.shape}")
print(f"Shape of the target data (y): {y.shape}")

print("\nSGD Model:")
print(f"Mean Absolute Error: {np.mean(np.abs(sgd_error)):.4f}")
print(f"Root Mean Squared Error: {np.sqrt(np.mean(sgd_error**2)):.4f}")

print("\nAdagrad Model:")
print(f"Mean Absolute Error: {np.mean(np.abs(adagrad_error)):.4f}")
print(f"Root Mean Squared Error: {np.sqrt(np.mean(adagrad_error**2)):.4f}")

# Calculate and print normalized RMSE
sgd_nrmse = np.sqrt(np.mean(sgd_error**2)) / (target.max() - target.min())
adagrad_nrmse = np.sqrt(np.mean(adagrad_error**2)) / (target.max() - target.min())

print("\nNormalized RMSE:")
print(f"SGD Model: {sgd_nrmse:.4f}")
print(f"Adagrad Model: {adagrad_nrmse:.4f}")
