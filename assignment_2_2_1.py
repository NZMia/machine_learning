import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Read the CSV file
data = pd.read_csv('Heat_Influx_insulation_east_south_north.csv')

# Separate the dataset into input and output
X = data[['Insulation', 'East', 'South', 'North']]
y = data['HeatFlux'].values.reshape(-1, 1)

# Normalize the data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_normalized = scaler_X.fit_transform(X)
y_normalized = scaler_y.fit_transform(y)

# Convert normalized data back to DataFrame for easier manipulation
X_normalized_df = pd.DataFrame(X_normalized, columns=X.columns)
y_normalized_df = pd.DataFrame(y_normalized, columns=['HeatFlux'])

# Plot all variables in the same plot
plt.figure(figsize=(12, 6))
plt.plot(X_normalized_df.index, y_normalized_df['HeatFlux'], label='HeatFlux', linewidth=2)
plt.plot(X_normalized_df.index, X_normalized_df['Insulation'], label='Insulation')
plt.plot(X_normalized_df.index, X_normalized_df['East'], label='East')
plt.plot(X_normalized_df.index, X_normalized_df['South'], label='South')
plt.plot(X_normalized_df.index, X_normalized_df['North'], label='North')

plt.xlabel('Data Point Index')
plt.ylabel('Normalized Value')
plt.title('HeatFlux and Input Variables')
plt.legend()
plt.grid(True)
plt.show()