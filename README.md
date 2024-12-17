The training dataset consists of 50 random samples drawn from a 2D input space, and the
corresponding outputs are generated using the function y = sin(2πx1) + cos(2πx2) with added
Gaussian noise. For testing, 20 new samples were used to assess the model’s predictions. The
kernel employed is a combination of the Radial Basis Function (RBF) and a white noise kernel.
The main steps of the implementation include:
1. Data Generation: Generate 2D training and test datasets with a nonlinear function.
2. Model Fitting: Fit a GPR model using the training dataset.
3. Prediction and Evaluation: Predict the mean and variance for the test data and compute
error metrics.
4. Visualization: Plot the results in both 3D and 2D spaces.
