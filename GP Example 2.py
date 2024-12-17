import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

# Generate training data
np.random.seed(42)
N = 50  # Number of training points
x_train = np.random.uniform(-1, 1, (N, 2))  # 2D input
y_train = np.sin(2 * np.pi * x_train[:, 0]) + np.cos(2 * np.pi * x_train[:, 1]) + np.random.normal(0, 0.1, N)

# Generate test data
M = 20  # Number of test points
x_test = np.random.uniform(-1, 1, (M, 2))  # 2D input for test data
y_real = np.sin(2 * np.pi * x_test[:, 0]) + np.cos(2 * np.pi * x_test[:, 1])  # Real values for test points

# Define the kernel: RBF + WhiteKernel for noise
length_scale = 1.0
noise_variance = 0.1
kernel = RBF(length_scale=length_scale) + WhiteKernel(noise_level=noise_variance)

# Create Gaussian Process Regressor
gp = GaussianProcessRegressor(kernel=kernel)

# Fit the model to the training data
gp.fit(x_train, y_train)

# Predict the mean and variance for the test points
mu_star, sigma_star = gp.predict(x_test, return_std=True)

# Compute errors
absolute_errors = np.abs(mu_star - y_real)
squared_errors = (mu_star - y_real) ** 2

# Compute average errors
average_absolute_error = np.mean(absolute_errors)
average_squared_error = np.mean(squared_errors)

# Print results
print(f"Average absolute error: {average_absolute_error:.3f}")
print(f"Average squared error: {average_squared_error:.3f}")

# Visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot training data
ax.scatter(x_train[:, 0], x_train[:, 1], y_train, color='red', label='Training Data')

# Plot test data with predictions
ax.scatter(x_test[:, 0], x_test[:, 1], mu_star, color='blue', label='Predicted Values')
ax.scatter(x_test[:, 0], x_test[:, 1], y_real, color='green', label='Real Values')

ax.set_title("Gaussian Process Regression with sklearn")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("y")
plt.legend()
plt.show()
# Sort the test data for better visualization
sorted_indices = np.argsort(y_real)
y_real_sorted = y_real[sorted_indices]
mu_star_sorted = mu_star[sorted_indices]

# Plot real vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(y_real_sorted, label="Real Values (y_real)", color="green", marker="o", linestyle="dashed")
plt.plot(mu_star_sorted, label="Predicted Values (mu_star)", color="blue", marker="x", linestyle="dashed")
plt.title("Real vs Predicted Values")
plt.xlabel("Test Data Points (sorted by y_real)")
plt.ylabel("Value")
plt.legend()
plt.grid()
plt.show()
# Scatter plot of real vs predicted values
plt.figure(figsize=(8, 8))
plt.scatter(y_real, mu_star, color="blue", alpha=0.6, label="Predicted vs Real")

# Plot the perfect prediction line (y = x)
plt.plot([y_real.min(), y_real.max()], [y_real.min(), y_real.max()], color="red", linestyle="--", label="Perfect Prediction (y = x)")

# Add labels, title, and legend
plt.title("Real Values vs Predicted Values")
plt.xlabel("Real Values (y_real)")
plt.ylabel("Predicted Values (mu_star)")
plt.legend()
plt.grid()
plt.axis("equal")  # Ensures square aspect ratio for proper comparison
plt.show()