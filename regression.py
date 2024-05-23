import numpy as np
import matplotlib.pyplot as plt

from data import get_data, inspect_data, split_data

data = get_data()
inspect_data(data)

train_data, test_data = split_data(data)

# Simple Linear Regression
# predict MPG (y, dependent variable) using Weight (x, independent variable) using closed-form solution
# y = theta_0 + theta_1 * x - we want to find theta_0 and theta_1 parameters that minimize the prediction error

# We can calculate the error using MSE metric:
# MSE = SUM (from i=1 to n) (actual_output - predicted_output) ** 2

# get the columns
y_train = train_data['MPG'].to_numpy()
x_train = train_data['Weight'].to_numpy()

y_test = test_data['MPG'].to_numpy()
x_test = test_data['Weight'].to_numpy()

# TODO: calculate closed-form solution
theta_best = [0, 0]

# TODO: calculate error
X_train = np.c_[np.ones((x_train.shape[0], 1)), x_train]
X_test = np.c_[np.ones((x_test.shape[0], 1)), x_test]

# Calculate closed-form solution for the parameters theta
theta_best = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)

# Predictions using the calculated parameters
y_pred_train = X_train.dot(theta_best)
y_pred_test = X_test.dot(theta_best)

# Calculate MSE for training and test sets
mse_train = np.mean((y_train - y_pred_train)**2)
mse_test = np.mean((y_test - y_pred_test)**2)

# Output the results
print("Theta (parameters):", theta_best)
print("Training MSE:", mse_train)
print("Test MSE:", mse_test)


plt.scatter(x_test, y_test, color='blue', label='Actual data')
plt.plot(x_test, y_pred_test, color='red', label='Regression line')
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.legend()
plt.show()

# Standardization (Z-score normalization) dostosowanie cech do wspólnej skali
mean = np.mean(x_train, axis=0)
std = np.std(x_train, axis=0)

x_train_standardized = (x_train - mean) / std
x_test_standardized = (x_test-mean)/std

# Adding a column of ones for the intercept term to x_train and x_test after standardization
X_train_std = np.c_[np.ones((x_train_standardized.shape[0], 1)), x_train_standardized]
X_test_std = np.c_[np.ones((x_test_standardized.shape[0], 1)), x_test_standardized]

# Batch Gradient Descent settings
learning_rate = 0.01
n_iterations = 1000
m = X_train_std.shape[0]

theta = np.random.randn(2, 1)  # Random initialization

#Gradient wskazuje kierunek najszybszego wzrostu funkcji kosztu, a więc, aby znaleźć minimum, należy poruszać się w przeciwnym kierunku. Wyrażenie X_train_std.dot(theta) - y_train.reshape(-1, 1) oblicza różnicę między wartościami przewidzianymi przez model a rzeczywistymi wartościami zmiennej zależnej
for iteration in range(n_iterations):
    gradients = 2/m * X_train_std.T.dot(X_train_std.dot(theta) - y_train.reshape(-1, 1))#zmień kształt y_train na macierz, która ma 1 kolumnę, a liczba wierszy niech odpowiada długości danych
    theta -= learning_rate * gradients

# Predictions using the calculated parameters
y_pred_train_std = X_train_std.dot(theta)
y_pred_test_std = X_test_std.dot(theta)

# Calculate MSE for training and test sets falatten bo chce wektor a nie wielowymiaryowy
mse_train_std = np.mean((y_train - y_pred_train_std.flatten())**2)
mse_test_std = np.mean((y_test - y_pred_test_std.flatten())**2)

# Output the results
print("Theta (parameters) after Gradient Descent:", theta.flatten())
print("Training MSE after standardization:", mse_train_std)
print("Test MSE after standardization:", mse_test_std)

# Plotting the regression line with test data
plt.scatter(x_test_standardized, y_test, color='blue', label='Actual data')
x_line = np.linspace(min(x_test_standardized), max(x_test_standardized), 100)
y_line = theta[0] + theta[1] * x_line
plt.plot(x_line, y_line, color='red', label='Regression line')
plt.xlabel('Standardized Weight')
plt.ylabel('MPG')
plt.legend()
plt.show()