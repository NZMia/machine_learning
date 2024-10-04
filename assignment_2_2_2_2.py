import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras

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

def create_model(learning_rate, momentum, hidden_neurons):
    model = keras.Sequential([
        keras.layers.Dense(hidden_neurons, activation='sigmoid', input_shape=(4,)),
        keras.layers.Dense(1, activation='linear')
    ])
    optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
    return model

def get_initial_weights(hidden_neurons):
    a0, a1 = 0.301843, 0.202268
    b0, b1 = -0.080305, 0.412510
    
    hidden_weights = np.tile([[a0], [a1]], (2, hidden_neurons))
    hidden_bias = np.full((hidden_neurons,), b0)
    output_weights = np.full((hidden_neurons, 1), b1)
    output_bias = np.array([0])
    
    return [hidden_weights, hidden_bias, output_weights, output_bias]

trials = [
    {'name': 'A', 'learning_rate': 0.1, 'momentum': 0.1},
    {'name': 'B', 'learning_rate': 0.1, 'momentum': 0.9},
    {'name': 'C', 'learning_rate': 0.5, 'momentum': 0.5},
    {'name': 'D', 'learning_rate': 0.9, 'momentum': 0.1},
    {'name': 'E', 'learning_rate': 0.9, 'momentum': 0.9},
]

hidden_neuron_configs = [3, 5]

all_results = []

for hidden_neurons in hidden_neuron_configs:
    print(f"\nRunning trials with {hidden_neurons} hidden neuron(s)")
    initial_weights = get_initial_weights(hidden_neurons)
    
    plt.figure(figsize=(15, 10))
    plt.suptitle(f'Performance for {hidden_neurons} Hidden Neuron(s)')
    
    for i, trial in enumerate(trials):
        model = create_model(trial['learning_rate'], trial['momentum'], hidden_neurons)
        model.set_weights(initial_weights)
        

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
        all_results.append({
            'Hidden Neurons': hidden_neurons,
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
        plt.subplot(2, 3, i+1)
        plt.plot(history.history['loss'], label='Training Error')
        plt.plot(history.history['val_loss'], label='Validation Error')
        plt.title(f"Trial {trial['name']}: LR={trial['learning_rate']}, M={trial['momentum']}")
        plt.xlabel('Epochs')
        plt.ylabel('Mean Squared Error')
        plt.legend()
        
        # y-axis normalization
        # plt.ylim(0, 0.3)

    plt.tight_layout()
    plt.show()


# Convert results to DataFrame and display
results_df = pd.DataFrame(all_results)
print(results_df.to_string(index=False))

# Find the best model based on test MSE and R²
best_models = results_df.groupby('Hidden Neurons').apply(lambda x: x.loc[x['Test MSE'].idxmin()])
print("\nBest Models for each Hidden Neuron configuration based on Test MSE:")
print(best_models.to_string(index=False))

# Select the overall best model
overall_best_model = best_models.loc[best_models['Test MSE'].idxmin()]
print("\nOverall Best Model:")
print(overall_best_model.to_string(index=False))

# Check if the best model also performs well on all data
all_data_performance = results_df.loc[results_df['All MSE'].idxmin()]
print("\nBest Model on All Data:")
print(all_data_performance.to_string(index=False))

if overall_best_model['Hidden Neurons'] == all_data_performance['Hidden Neurons'] and \
   overall_best_model['Trial'] == all_data_performance['Trial']:
    print("\nThe best model on test data also performs best on all data.")
else:
    print("\nThe best model on test data is different from the best model on all data.")
    print("Consider the trade-off between performance on test data and all data.")