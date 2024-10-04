import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from keras import layers
# Plot training history
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
# np.random.seed(0)
# tf.random.set_seed(0)

# Load and preprocess the data
data = pd.read_csv('Heat_Influx_insulation_east_south_north.csv')
X = data[['Insulation', 'East', 'South', 'North']]
y = data['HeatFlux'].values.reshape(-1, 1)

# Normalize the data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_normalized = scaler_X.fit_transform(X)
y_normalized = scaler_y.fit_transform(y)

# Split the dataset into train_val (80%) and test (20%)
X_train_val, X_test, y_train_val, y_test = train_test_split(X_normalized, y_normalized, test_size=0.2, random_state=42)
# Split train_val into train (75%) and validation (25%)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

print(f'Training set size: {X_train.shape[0]}')
print(f'Validation set size: {X_val.shape[0]}')
print(f'Test set size: {X_test.shape[0]}')

model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)), # Input layer layers.Dense(32, activation='relu'), # Hidden layer
    layers.Dense(1) # Output layer
])
# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
# training dataset for 500 epochs
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=500, batch_size=32, verbose=1)
# Evaluate the model on the test set
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=2)
print(f'Test Loss: {test_loss}, Test MAE: {test_mae}')
# Make predictions
predictions = model.predict(X_test)

predictions_inverse = scaler_y.inverse_transform(predictions) 
y_test_inverse = scaler_y.inverse_transform(y_test)

# Compare predictions with actual values
plt.figure(figsize=(10, 6))
plt.plot(y_test_inverse, label='Actual HeatFlux', color='blue') 
plt.plot(predictions_inverse, label='Predicted HeatFlux', color='red') 
plt.title('Actual vs Predicted HeatFlux')
plt.xlabel('Sample Index')
plt.ylabel('HeatFlux')
plt.legend()
plt.show()