

# Version 3
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Read the CSV file
data = pd.read_csv('Heat_Influx_insulation_east_south_north.csv')

# Define input and output variables
input_vars = ['Insulation', 'East', 'South', 'North']
output_var = 'HeatFlux'

# Normalize the dataset using min-max scaling
scaler = MinMaxScaler()
normalized_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

# Use only the first 30 datapoints
normalized_data = normalized_data.head(30)

# Create a figure with 4 subplots
fig, axs = plt.subplots(2, 2, figsize=(15, 15))
fig.suptitle('HeatFlux vs Input Variables (First 30 Datapoints)', fontsize=16)

# Plot each input variable against HeatFlux
for i, var in enumerate(input_vars):
    row = i // 2
    col = i % 2
    axs[row, col].plot(normalized_data.index, normalized_data[output_var], label=[output_var], color='red')
    axs[row, col].plot(normalized_data.index, normalized_data[var], label=var, color='blue')
    axs[row, col].set_title(f'HeatFlux vs {var}')
    axs[row, col].set_xlabel('Data Point Index')
    axs[row, col].set_ylabel('Normalized Value')
    axs[row, col].legend()
    axs[row, col].grid(True)

plt.tight_layout()
plt.show()