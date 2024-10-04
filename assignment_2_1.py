import numpy as np

# Initial weights after first epoch
a0 = 0.301444
a1 = 0.201954
b0 = -0.0844103
b1 = 0.409993

# Learning rate
beta = 0.1

# Input and target values
inputs = np.array([0.7853, 1.57])
targets = np.array([0.707, 1.0])

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Forward pass: calculate the network output
def forward(x, a0, a1, b0, b1):
    u = a0 + a1 * x
    y = sigmoid(u)
    v = b0 + b1 * y
    z = sigmoid(v)
    return u, y, v, z

# Compute the gradients
def compute_gradients(x, t, u, y, v, z, b1):
    dz = (z - t) * z * (1 - z)  # dz/dv
    db0 = dz
    db1 = dz * y
    dy = dz * b1 * y * (1 - y)  # dy/du
    da0 = dy
    da1 = dy * x
    print(f"db1: {db1}")
    print(f"db0: {db0}")
    print(f"da1: {da1}")
    print(f"da0: {da0}")
    return db0, db1, da0, da1

# Update weights
def update_weights(a0, a1, b0, b1, db0, db1, da0, da1, beta):
    a0_new = a0 - beta * da0
    a1_new = a1 - beta * da1
    b0_new = b0 - beta * db0
    b1_new = b1 - beta * db1
    return a0_new, a1_new, b0_new, b1_new

# Perform weight adjustment for first input (0.7853)
x1 = inputs[0]
t1 = targets[0]

# Forward pass for input 1
u1, y1, v1, z1 = forward(x1, a0, a1, b0, b1)

# Compute gradients for input 1
db0_1, db1_1, da0_1, da1_1 = compute_gradients(x1, t1, u1, y1, v1, z1, b1)

# Update weights after input 1
a0_new, a1_new, b0_new, b1_new = update_weights(a0, a1, b0, b1, db0_1, db1_1, da0_1, da1_1, beta)

# Calculate network output for both inputs with updated weights
outputs = []
for x in inputs:
    _, _, _, z = forward(x, a0_new, a1_new, b0_new, b1_new)
    outputs.append(z)

# Mean Squared Error (MSE) for both inputs
outputs = np.array(outputs)
mse = np.mean((outputs - targets) ** 2)

# Print the results
print(f"Updated weights: a0'={a0_new}, a1'={a1_new}, b0'={b0_new}, b1'={b1_new}")
print(f"Outputs for inputs {inputs}: {outputs}")
print(f"Mean Squared Error: {mse}")