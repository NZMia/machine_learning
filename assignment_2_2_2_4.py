import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras

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

# 1 is the number of neurons in my best model from previous experiments
best_hidden_neurons = 3

def create_adagrad_model(hidden_neurons):
    model = keras.Sequential([
        keras.layers.Dense(hidden_neurons, activation='sigmoid', input_shape=(4,)),
        keras.layers.Dense(1, activation='linear')
    ])
    # Using default learning rate
    optimizer = keras.optimizers.Adagrad()  
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
    return model

# Train the model
model = create_adagrad_model(best_hidden_neurons)
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=500, batch_size=32, verbose=1)


# Evaluate the model
def evaluate_model(model, X, y):
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    r2 = r2_score(y, predictions)
    return mse, r2

train_mse, train_r2 = evaluate_model(model, X_train, y_train)
val_mse, val_r2 = evaluate_model(model, X_val, y_val)
test_mse, test_r2 = evaluate_model(model, X_test, y_test)
all_mse, all_r2 = evaluate_model(model, X_normalized, y_normalized)

# Print results
print(f"Number of Neurons in Hidden Layer: {best_hidden_neurons}")
print("MSE:")
print(f"Train: {train_mse:.6f}, Validation: {val_mse:.6f}, Test: {test_mse:.6f}, All: {all_mse:.6f}")
print("R2:")
print(f"Train: {train_r2:.6f}, Validation: {val_r2:.6f}, Test: {test_r2:.6f}, All: {all_r2:.6f}")

# Plot training history
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Save the model
model.save('best_adagrad_model.h5')
print("Adagrad model saved as 'best_adagrad_model.h5'")