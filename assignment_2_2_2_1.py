import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras

# # Set random seeds for reproducibility
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

def create_model(learning_rate, momentum):
    model = keras.Sequential([
        # Hidden layer: 1 neuron with Sigmoid activation
        keras.layers.Dense(1, activation='sigmoid', input_shape=(4,)),
        # Output layer: 1 neuron with Linear activation
        keras.layers.Dense(1, activation='linear')
    ])
    optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
    return model

# Define initial weights as same as part one
a0, a1 = 0.301843, 0.202268
b0, b1 = -0.080305, 0.412510

hidden_weights = np.tile([[a0], [a1]], (2, 1))
hidden_bias = np.full((1,), b0)
output_weights = np.full((1, 1), b1)
output_bias = np.array([0])

initial_weights = [hidden_weights, hidden_bias, output_weights, output_bias]

# Define trials
trials = [
    {'name': 'A', 'learning_rate': 0.1, 'momentum': 0.1},
    {'name': 'B', 'learning_rate': 0.1, 'momentum': 0.9},
    {'name': 'C', 'learning_rate': 0.5, 'momentum': 0.5},
    {'name': 'D', 'learning_rate': 0.9, 'momentum': 0.1},
    {'name': 'E', 'learning_rate': 0.9, 'momentum': 0.9},
]

results = []

for trial in trials:
    # Create and compile the model
    model = create_model(trial['learning_rate'], trial['momentum'])
    model.set_weights(initial_weights)
    
    # Train the model
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=500, batch_size=32, verbose=0)
    
    # Evaluate the model
    train_predictions = model.predict(X_train)
    val_predictions = model.predict(X_val)
    test_predictions = model.predict(X_test)
    all_predictions = model.predict(X_normalized)
    
    # Calculate MSE and R²
    train_mse = mean_squared_error(y_train, train_predictions)
    val_mse = mean_squared_error(y_val, val_predictions)
    test_mse = mean_squared_error(y_test, test_predictions)
    all_mse = mean_squared_error(y_normalized, all_predictions)
    
    train_r2 = r2_score(y_train, train_predictions)
    val_r2 = r2_score(y_val, val_predictions)
    test_r2 = r2_score(y_test, test_predictions)
    all_r2 = r2_score(y_normalized, all_predictions)
    
    # Store results
    results.append({
        'Trial': trial['name'],
        'Learning Rate': trial['learning_rate'],
        'Momentum': trial['momentum'],
        'Train MSE': train_mse,
        'Validation MSE': val_mse,
        'Test MSE': test_mse,
        'All MSE': all_mse,
        'Train R²': train_r2,
        'Validation R²': val_r2,
        'Test R²': test_r2,
        'All R²': all_r2
    })
    
    # Plot training and validation error
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Error')
    plt.plot(history.history['val_loss'], label='Validation Error')
    plt.title(f"Trial {trial['name']}: Error over Epochs\nLR: {trial['learning_rate']}, Momentum: {trial['momentum']}")
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    
    # Plot training and validation accuracy (using MAE as a proxy for accuracy)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training Accuracy')
    plt.plot(history.history['val_mae'], label='Validation Accuracy')
    plt.title(f"Trial {trial['name']}: Accuracy over Epochs\nLR: {trial['learning_rate']}, Momentum: {trial['momentum']}")
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Convert results to DataFrame and display
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# Find the best model based on test MSE
best_model = results_df.loc[results_df['Test MSE'].idxmin()]
print("\nBest Model based on Test MSE:")
print(best_model.to_string(index=False))

# Check if the best model also performs well on the whole dataset
if best_model['All MSE'] == results_df['All MSE'].min():
    print("\nThe best model based on Test MSE also performs best on the whole dataset.")
else:
    print("\nThe best model based on Test MSE is not the best performer on the whole dataset.")
    print("Best model for 30 dataset:")
    print(results_df.loc[results_df['All MSE'].idxmin()].to_string(index=False))