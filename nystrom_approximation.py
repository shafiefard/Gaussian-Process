import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.metrics.pairwise import rbf_kernel
# Nyström approximation function
def nystrom_approximation(X, m, gamma,y):
    """
    X: Input data (n x d)
    m: Number of inducing points (subset size)
    gamma: Kernel coefficient for RBF
    """
    n = X.shape[0]
    # Step 1: Randomly sample m inducing points
    indices = np.random.choice(n, m, replace=False)
    X_m = X[indices]
    y_m = y[indices]

    # Step 2: Compute kernel matrices
    K_mm = rbf_kernel(X_m, X_m, gamma=gamma)  # Small kernel (m x m)
    K_nm = rbf_kernel(X, X_m, gamma=gamma)    # Cross-kernel (n x m)

    # Step 3: Approximate full kernel
    f_pred = K_nm @ np.linalg.pinv(K_mm)@ y_m
    K_approx = K_nm @ np.linalg.pinv(K_mm) @ K_nm.T
    return K_approx, f_pred, X_m, y_m

# Generate synthetic data
np.random.seed(42)
N = 200  # Total number of data points
x = np.linspace(-3, 3, N).reshape(-1, 1)  # 1D input
#y = np.sin(x).ravel() + np.log(x**2).ravel()+np.random.normal(0, 0.2, N)  # Noisy sine wave
y = 10*np.sin(4*x).ravel() + 10*np.sin(7*x).ravel() + np.random.normal(0, 0.2, N)  # Noisy sine wave

# Apply Nyström approximation
gamma = 4  # RBF kernel parameter
m = 25      # Subset size for Nyström approximation
K_nystrom, y_pred, X_m, y_m = nystrom_approximation(x, m, gamma,y)

# Visualization
plt.figure(figsize=(10, 6))
plt.plot(x, y, "r.", label="Noisy Training Data")
plt.plot(x, y_pred, "b-", label="Nyström Approximation Prediction")
plt.title("Gaussian Process Regression with Nyström Approximation")
plt.scatter(X_m, y_m, color="orange", label="Inducing points", s=50)
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()
