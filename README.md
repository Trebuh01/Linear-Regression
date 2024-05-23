# Project Title: Simple Linear Regression and Standardization with Auto MPG Dataset

## Overview

This project demonstrates the implementation of a simple linear regression model to predict the miles per gallon (MPG) of cars based on their weight using both the closed-form solution and gradient descent. Additionally, it includes standardization of features to improve the performance of the gradient descent algorithm. The Auto MPG dataset from the UCI Machine Learning Repository is used for this purpose.

## Data Exploration and Preprocessing

The `data.py` file contains functions to handle data loading, inspection, and splitting:

- `get_data()`: Loads the dataset, removes entries with missing values, and returns the cleaned dataset.
- `inspect_data(dataset)`: Displays the dataset's shape, tail, and statistical summary. It also plots pairwise relationships.
- `split_data(dataset)`: Splits the dataset into training (80%) and testing (20%) sets.

## Simple Linear Regression

The main script (`main.py`) performs simple linear regression using the closed-form solution to predict MPG based on car weight:

- **Extract Columns**: Extracts the MPG and Weight columns from the training and testing datasets.
- **Closed-Form Solution**: Calculates the optimal parameters (theta) that minimize the mean squared error (MSE).
- **Prediction and MSE Calculation**: Computes predictions on the training and testing sets and calculates the MSE for both.

## Standardization and Gradient Descent

The script also includes feature standardization and implements batch gradient descent:

- **Standardization**: Standardizes the weight feature using z-score normalization.
- **Batch Gradient Descent**: Optimizes the parameters using gradient descent over a set number of iterations and learning rate.
- **Prediction and MSE Calculation**: Computes predictions on the standardized training and testing sets and calculates the MSE for both.

## Results

The script outputs the following results:

- Parameters (theta) for the closed-form solution and after gradient descent.
- MSE for training and testing sets before and after standardization.
- Plots showing the regression line over the actual data points for both raw and standardized features.

