import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics.pairwise import rbf_kernel

# Nyström approximation function
def nystrom_approximation(X, m, gamma, y):
    """
    X: Input data (n x d)
    m: Number of inducing points (subset size)
    gamma: Kernel coefficient for RBF
    y: Target values (n,)
    """
    n = X.shape[0]
    assert y.shape[0] == n, "Mismatch between the number of rows in X and y."

    # Step 1: Randomly sample m inducing points
    indices = np.random.choice(n, m, replace=False)
    X_m = X[indices]
    y_m = y[indices]

    # Step 2: Compute kernel matrices
    K_mm = rbf_kernel(X_m, X_m, gamma=gamma)  # Small kernel (m x m)
    K_nm = rbf_kernel(X, X_m, gamma=gamma)    # Cross-kernel (n x m)

    # Step 3: Approximate full kernel and predictions
    f_pred = K_nm @ np.linalg.pinv(K_mm) @ y_m
    K_approx = K_nm @ np.linalg.pinv(K_mm) @ K_nm.T
    return K_approx, f_pred, X_m, y_m

# Generate synthetic 2D data
np.random.seed(42)
N = 200  # Total number of data points
x1 = np.linspace(-3, 3, N).reshape(-1, 1)  # First dimension
x2 = np.linspace(-3, 3, N).reshape(-1, 1)  # Second dimension
X = np.hstack((x1, x2))  # Combine into 2D input

# Define target function with noise
y = 10 * np.sin(4 * x1).ravel() + 5 * np.cos(3 * x2).ravel()
y += np.random.normal(0, 0.2, N)  # Add noise

# Apply Nyström approximation
gamma = 4  # RBF kernel parameter
m = 25     # Subset size for Nyström approximation
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)
K_nystrom, y_pred, X_m, y_m = nystrom_approximation(X, m, gamma, y)

# Visualization in 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for noisy data
ax.scatter(X[:, 0], X[:, 1], y, c='r', marker='o', label="Noisy Training Data", alpha=0.5)

# Create a grid for predictions
x1_grid, x2_grid = np.meshgrid(np.linspace(-3, 3, 50), np.linspace(-3, 3, 50))
X_grid = np.hstack((x1_grid.reshape(-1, 1), x2_grid.reshape(-1, 1)))

# Compute kernel-based predictions for X_grid
K_nm_grid = rbf_kernel(X_grid, X_m, gamma=gamma)
y_grid = K_nm_grid @ np.linalg.pinv(rbf_kernel(X_m, X_m, gamma=gamma)) @ y_m
y_grid = y_grid.reshape(x1_grid.shape)

# Surface plot for Nyström approximation predictions
ax.plot_surface(x1_grid, x2_grid, y_grid, color='blue', alpha=0.7)

# Highlight inducing points
ax.scatter(X_m[:, 0], X_m[:, 1], y_m, c='orange', s=50, label="Inducing Points")

# Labels and legend
ax.set_title("Gaussian Process Regression with Nyström Approximation (2D Input)")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("y")
ax.legend()
plt.show()
