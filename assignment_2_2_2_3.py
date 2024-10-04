import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

# Set best parameters from previous runs
best_learning_rate = 0.9 
best_momentum = 0.9 
best_hidden_neurons = 3

# Set random seeds for reproducibility
np.random.seed(0)
tf.random.set_seed(0)

# Load and preprocess the data
data = pd.read_csv('Heat_Influx_insulation_east_south_north.csv')
X = data[['Insulation', 'East', 'South', 'North']]
y = data['HeatFlux'].values.reshape(-1, 1)

# Normalize the data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_normalized = scaler_X.fit_transform(X)
y_normalized = scaler_y.fit_transform(y)

# Split the data
X_train_val, X_test, y_train_val, y_test = train_test_split(X_normalized, y_normalized, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

# Recreate the best model
def create_best_model():
    model = keras.Sequential([
        keras.layers.Dense(best_hidden_neurons, activation='sigmoid', input_shape=(4,)),
        keras.layers.Dense(1, activation='linear')
    ])
    optimizer = keras.optimizers.SGD(learning_rate=best_learning_rate, momentum=best_momentum)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
    return model

# Create and train the best model
best_model = create_best_model()
history = best_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=500, batch_size=32, verbose=1)

# Save the model
best_model.save('best_heat_influx_model.h5')


# Evaluate the model
test_predictions = best_model.predict(X_test)
test_mse = mean_squared_error(y_test, test_predictions)
test_r2 = r2_score(y_test, test_predictions)

# Save results to a file
with open('model_results.txt', 'w') as f:
    f.write(f"Best Model Configuration:\n")
    f.write(f"Hidden Neurons: {best_hidden_neurons}\n")
    f.write(f"Learning Rate: {best_learning_rate}\n")
    f.write(f"Momentum: {best_momentum}\n")
    f.write(f"Test MSE: {test_mse}\n")
    f.write(f"Test R²: {test_r2}\n")

print("Model saved with initial weights and results exported.")